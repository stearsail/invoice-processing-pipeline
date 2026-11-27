export const CATEGORY_STYLES = {
  unverifiable: 'text-amber-300',
  mismatch: 'text-red-300',
  duplicate: 'text-purple-300',
}

// Plain-language copy for the "category: detail" prefix categorize_error() puts on each parse error
export const ERROR_CATEGORY_MESSAGES = {
  connectivity:
    'Temporarily unable to reach the extraction service — this usually resolves on retry.',
  bad_input: 'Bad input, error decoding or identifying image file.',
  rejected_input: 'The extraction service rejected this document.',
  invalid_output:
    "The extracted data didn't match the expected format for one or more fields (e.g., a non-numeric value where a number was expected).",
  unknown: 'Something unexpected happened while processing this document.',
}

// job.error on a total failure joins every attempt's parse error with "; "
export function friendlyJobError(error) {
  if (!error) return null
  const messages = Object.keys(ERROR_CATEGORY_MESSAGES)
    .filter((category) => error.includes(`${category}:`))
    .map((category) => ERROR_CATEGORY_MESSAGES[category])
  return messages.length > 0 ? messages.join(' ') : error
}

// Receipts don't carry customer or due-date info in this schema's training
// data — customer/due_date stay valid null fields either way, this only
// controls what the UI asks a reviewer to look at.
export function isReceipt(documentType) {
  return documentType === 'receipt'
}

// An attribution line, not a warning; null when no model produced an invoice
export function extractionSourceLabel(extractor, corrected, { compact = false } = {}) {
  if (!extractor) return null
  if (compact) {
    return corrected ? `${extractor} · edited` : extractor
  }
  return corrected
    ? `Extracted by ${extractor} · corrected on review`
    : `Extracted by ${extractor}`
}

// Once approved, the model's issues and errors are resolved history, not open problems
export function toReviewInfo(item) {
  const approved = item.ledger_entry != null
  return {
    sourceLabel: extractionSourceLabel(item.extraction?.extractor, item.ledger_entry?.corrected),
    jobError: approved ? null : item.error,
    issues: approved ? [] : (item.extraction?.issues ?? []),
    loadErrors: [item.extraction_error, item.ledger_entry_error].filter(Boolean),
  }
}

// An edit continues from the approved invoice; a first review starts from the model's
export function startingInvoice(item) {
  return toFormInvoice(item.ledger_entry?.invoice_data ?? item.extraction?.invoice_data)
}

export const BLANK_PARTY = { name: '', address: '', tax_id: '', iban: '' }
export const BLANK_LINE_ITEM = {
  description: '',
  unit_price: '',
  quantity: '',
  line_total: '',
  marked: false,
}

export function lineItemHasData(line) {
  return (
    line.description.trim() !== '' ||
    String(line.unit_price).trim() !== '' ||
    String(line.quantity).trim() !== '' ||
    String(line.line_total).trim() !== ''
  )
}

export const BLANK_INVOICE = {
  document_type: 'invoice',
  vendor: { ...BLANK_PARTY },
  customer: { ...BLANK_PARTY },
  invoice_number: '',
  issue_date: '',
  due_date: '',
  currency: '',
  line_items: [{ ...BLANK_LINE_ITEM }],
  grand_total: '',
  subtotal: '',
  tax: '',
  service_charge: '',
  discount: '',
  confidence_notes: [],
}

export function toFormInvoice(invoiceData) {
  if (!invoiceData) return BLANK_INVOICE
  return {
    document_type: invoiceData.document_type ?? 'invoice',
    vendor: invoiceData.vendor
      ? { ...BLANK_PARTY, ...invoiceData.vendor }
      : { ...BLANK_PARTY },
    customer: invoiceData.customer
      ? { ...BLANK_PARTY, ...invoiceData.customer }
      : { ...BLANK_PARTY },
    invoice_number: invoiceData.invoice_number ?? '',
    issue_date: invoiceData.issue_date ?? '',
    due_date: invoiceData.due_date ?? '',
    currency: invoiceData.currency ?? '',
    line_items:
      invoiceData.line_items && invoiceData.line_items.length > 0
        ? invoiceData.line_items.map((line) => ({ ...BLANK_LINE_ITEM, ...line }))
        : [{ ...BLANK_LINE_ITEM }],
    grand_total: invoiceData.grand_total ?? '',
    subtotal: invoiceData.subtotal ?? '',
    tax: invoiceData.tax ?? '',
    service_charge: invoiceData.service_charge ?? '',
    discount: invoiceData.discount ?? '',
    confidence_notes: invoiceData.confidence_notes ?? [],
  }
}