from typing import Annotated, Literal
from datetime import date
from decimal import Decimal
from pydantic import BaseModel, ConfigDict, Field, WithJsonSchema, field_validator


class Party(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    address: str | None = None
    tax_id: str | None = None
    iban: str | None = None


class LineItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    description: str | None = None
    unit_price: Decimal | None = None
    quantity: Decimal | None = None
    line_total: Decimal | None = None


class Invoice(BaseModel):
    model_config = ConfigDict(extra="forbid")
    vendor: Party | None = None
    customer: Party | None = None
    document_type: Literal["invoice", "receipt"]
    invoice_number: str | None = None
    issue_date: date | None = None
    due_date: date | None = None
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    line_items: list[LineItem] = Field(default_factory=list)
    grand_total: Decimal
    subtotal: Decimal | None = None
    tax: Decimal | None = None
    service_charge: Decimal | None = None
    discount: Decimal | None = None
    confidence_notes: list[str] = Field(default_factory=list)

    @field_validator("discount", mode="after")
    @classmethod
    def normalize_discount(cls, value: Decimal | None) -> Decimal | None:
        if value is None:
            return None
        return value.copy_abs()


# Wire schemas are needed for vLLM's guided decoding which use grammar-constrained decoding backend
# as Decimal fields use regex pattern that is negative lookahead and breaks compilation
# Annotated adds extra metadata to the type, overwriting the default model_json_schema fields
# while maintaining type at runtime
WireDecimal = Annotated[
    Decimal,
    WithJsonSchema({"type": "string", "pattern": r"^-?\d+(\.\d+)?$"}),
]


class WireLineItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    description: str | None = None
    unit_price: WireDecimal | None = None
    quantity: WireDecimal | None = None
    line_total: WireDecimal | None = None


class WireInvoice(BaseModel):
    model_config = ConfigDict(extra="forbid")
    vendor: Party | None = None
    customer: Party | None = None
    document_type: Literal["invoice", "receipt"]
    invoice_number: str | None = None
    issue_date: date | None = None
    due_date: date | None = None
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    line_items: list[WireLineItem] = Field(default_factory=list)
    grand_total: WireDecimal
    subtotal: WireDecimal | None = None
    tax: WireDecimal | None = None
    service_charge: WireDecimal | None = None
    discount: WireDecimal | None = None
    confidence_notes: list[str] = Field(default_factory=list)
