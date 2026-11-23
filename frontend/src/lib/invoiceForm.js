export const CATEGORY_STYLES = {
  unverifiable: 'text-amber-300',
  mismatch: 'text-red-300',
  duplicate: 'text-purple-300',
}

// Maps the category prefix categorize_error() embeds in "unverifiable" issue
// messages (e.g. "Extraction failed entirely — invalid_output: 1 validation
// error for Invoice ...") to plain-language copy for non-developers.
export const ERROR_CATEGORY_MESSAGES = {
  connectivity:
    'Temporarily unable to reach the extraction service — this usually resolves on retry.',
  bad_input: 'Bad input, error decoding or identifying image file.',
  invalid_output:
    "The extracted data didn't match the expected format for one or more fields (e.g., a non-numeric value where a number was expected).",
  unknown: 'Something unexpected happened while processing this document.',
}

// Only "unverifiable" issues carry an embedded category prefix (it's the one
// category validate_reconcile builds from a raw parse_error); other
// categories (mismatch, duplicate) already have human-authored messages.
export function friendlyIssueMessage(issue) {
  if (issue.category !== 'unverifiable') return issue.message
  const category = Object.keys(ERROR_CATEGORY_MESSAGES).find((cat) =>
    issue.message.includes(`${cat}:`)
  )
  return category ? ERROR_CATEGORY_MESSAGES[category] : issue.message
}

// Receipts don't carry customer or due-date info in this schema's training
// data — customer/due_date stay valid null fields either way, this only
// controls what the UI asks a reviewer to look at.
export function isReceipt(documentType) {
  return documentType === 'receipt'
}

// Deliberately understated — an attribution line, not a warning banner. Null
// for entries with no captured attempt (written by a human from scratch via
// edit_job, or predating this field).
export function extractionSourceLabel(extractedBy, reviewedWithChanges, { compact = false } = {}) {
  if (!extractedBy) return null
  if (compact) {
    return reviewedWithChanges ? `${extractedBy} · edited` : extractedBy
  }
  return reviewedWithChanges
    ? `Extracted by ${extractedBy} · corrected on review`
    : `Extracted by ${extractedBy}`
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