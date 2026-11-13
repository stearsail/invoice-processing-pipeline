import asyncio
from dataclasses import dataclass
import json
import logging
from invoice_pipeline.workflow.error_categories import categorize_error
from invoice_pipeline.workflow.images import load_image_b64
from invoice_pipeline.schema import Invoice, WireInvoice
from langchain_anthropic import ChatAnthropic
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    invoice: Invoice | None
    parse_error: str | None


class SpecialistExtractor:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model_name: str,
        *,
        timeout: float,
        max_retries: int = 0,
        seed: int | None = None,
    ):
        # max_retries defaults to 0 because in production arq owns retries —
        # SDK retries would stack on top of them and burn the job deadline
        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            max_retries=max_retries,
            timeout=timeout,
        )
        self._model_name = model_name
        self._seed = seed

    async def extract_invoice(self, img_path: str) -> ExtractionResult:
        response = None
        try:
            img_b64, mime_type = await asyncio.to_thread(load_image_b64, img_path)
            response = await self._client.chat.completions.create(
                model=self._model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Extract structured data from this receipt/invoice image as JSON.",
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{img_b64}"
                                },
                            },
                        ],
                    }
                ],
                # vllm-openai >= 0.26 silently ignores the legacy guided_json / guided_decoding_backend params
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "WireInvoice",
                        "schema": WireInvoice.model_json_schema(),
                        "strict": True,
                    },
                },
                seed=self._seed,
            )
            return ExtractionResult(
                invoice=Invoice.model_validate(
                    json.loads(response.choices[0].message.content)
                ),
                parse_error=None,
            )
        except Exception as e:
            category = categorize_error(e)
            if category == "connectivity":
                raise
            if response is not None:
                logger.warning(
                    "Extraction parse failure (%s): raw=%r",
                    category,
                    response.choices[0].message.content[:2000],
                )
            return ExtractionResult(invoice=None, parse_error=f"{category}: {e}")


class FrontierExtractor:
    def __init__(
        self,
        model_name: str,
        *,
        timeout: float,
        temperature: float | None = None,
        max_tokens: int = 4096,
    ):
        self._client = ChatAnthropic(
            model=model_name,
            temperature=temperature if temperature else None,
            max_tokens_to_sample=max_tokens,
            max_retries=0,
            timeout=timeout,
        )

    async def extract_invoice(
        self, img_path: str, extra_context: str = ""
    ) -> ExtractionResult:
        try:
            img_b64, mime_type = await asyncio.to_thread(load_image_b64, img_path)
            message = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """Extract structured data from this receipt/invoice image as JSON. 
                        Read the currency, number formatting, and locale directly from what's visible in the document 
                        — do not assume any specific country's conventions."""
                        + extra_context,
                    },
                    {
                        "type": "image",
                        "base64": img_b64,
                        "mime_type": mime_type,
                    },
                ],
            }
            structured_model = self._client.with_structured_output(
                Invoice, include_raw=True
            )
            result = await structured_model.ainvoke([message])
        except Exception as e:
            category = categorize_error(e)
            if category == "connectivity":
                raise
            return ExtractionResult(invoice=None, parse_error=f"{category}: {e}")
        if result["parsing_error"] is not None:
            category = categorize_error(result["parsing_error"])
            raw_text = result["raw"].content if result.get("raw") else None
            logger.warning(
                "Extraction parse failure (unknown): raw=%r", str(raw_text)[:2000]
            )
            return ExtractionResult(
                invoice=None, parse_error=f"unknown:{result['parsing_error']}"
            )
        return ExtractionResult(invoice=result["parsed"], parse_error=None)
