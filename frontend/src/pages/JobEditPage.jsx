import { useEffect, useState } from 'react'
import { useParams, useNavigate, Link } from 'react-router-dom'
import {
  getJobDetail,
  getReviewQueue,
  submitJobEdit,
  deleteJob,
  getJobImageUrl,
} from '../lib/api'
import ReviewIssues from '../components/ReviewIssues'
import {
  BLANK_LINE_ITEM,
  lineItemHasData,
  isReceipt,
  startingInvoice,
  toReviewInfo,
} from '../lib/invoiceForm'

function blankToNull(value) {
  return value === '' ? null : value
}

function toPayload(form) {
  const receipt = isReceipt(form.document_type)
  const vendor = form.vendor.name.trim()
    ? {
      name: form.vendor.name,
      address: blankToNull(form.vendor.address),
      tax_id: blankToNull(form.vendor.tax_id),
      iban: blankToNull(form.vendor.iban),
    }
    : null
  // customer/due_date are hidden on the receipt form — send null regardless
  // of what's still sitting in form state (e.g. left over from switching an
  // invoice's document_type to receipt mid-edit) rather than the stale value.
  const customer =
    !receipt && form.customer.name.trim()
      ? {
        name: form.customer.name,
        address: blankToNull(form.customer.address),
        tax_id: blankToNull(form.customer.tax_id),
        iban: blankToNull(form.customer.iban),
      }
      : null

  return {
    document_type: form.document_type,
    vendor,
    customer,
    invoice_number: blankToNull(form.invoice_number),
    issue_date: blankToNull(form.issue_date),
    due_date: receipt ? null : blankToNull(form.due_date),
    currency: form.currency.toUpperCase(),
    line_items: form.line_items
      .filter((line) => line.description.trim() !== '' && !line.marked)
      .map((line) => ({
        description: blankToNull(line.description),
        unit_price: blankToNull(line.unit_price),
        quantity: blankToNull(line.quantity),
        line_total: blankToNull(line.line_total),
      })),
    grand_total: form.grand_total,
    subtotal: blankToNull(form.subtotal),
    tax: blankToNull(form.tax),
    service_charge: blankToNull(form.service_charge),
    discount: blankToNull(form.discount),
    confidence_notes: form.confidence_notes,
  }
}

const inputClass =
  'w-full rounded border border-edge-strong bg-surface px-2 py-1 text-sm text-body focus:border-accent focus:outline-none'
const labelClass = 'block text-xs font-medium text-muted mb-1'

function Field({ label, ...props }) {
  return (
    <div>
      <label className={labelClass}>{label}</label>
      <input className={inputClass} {...props} />
    </div>
  )
}

export default function JobEditPage() {
  const { jobId } = useParams()
  return <JobEditPageInner key={jobId} />
}

function JobEditPageInner() {
  const { jobId } = useParams()
  const navigate = useNavigate()
  const [form, setForm] = useState(null)
  // load failures replace the page; save/delete failures (e.g. a 409) keep the form
  const [loadError, setLoadError] = useState(null)
  const [actionError, setActionError] = useState(null)
  const [submitting, setSubmitting] = useState(false)
  const [deleting, setDeleting] = useState(false)
  const [reviewInfo, setReviewInfo] = useState(null)
  const receipt = isReceipt(form?.document_type)

  useEffect(() => {
    async function load() {
      try {
        const item = await getJobDetail(jobId)
        setForm(startingInvoice(item))
        setReviewInfo(toReviewInfo(item))
      } catch (err) {
        if (err.status === 404) {
          setForm(false)
        } else {
          setLoadError(err.message)
        }
      }
    }
    load()
  }, [jobId])

  function updateField(key, value) {
    setForm((prev) => ({ ...prev, [key]: value }))
  }

  function updateParty(partyKey, field, value) {
    setForm((prev) => ({
      ...prev,
      [partyKey]: { ...prev[partyKey], [field]: value },
    }))
  }

  function updateLineItem(index, field, value) {
    setForm((prev) => ({
      ...prev,
      line_items: prev.line_items.map((line, i) =>
        i === index ? { ...line, [field]: value } : line
      ),
    }))
  }

  function toggleLineItemMarked(index) {
    setForm((prev) => ({
      ...prev,
      line_items: prev.line_items.map((line, i) =>
        i === index ? { ...line, marked: !line.marked } : line
      ),
    }))
  }

  function addLineItem() {
    setForm((prev) => ({
      ...prev,
      line_items: [...prev.line_items, { ...BLANK_LINE_ITEM }],
    }))
  }

  function removeLineItem(index) {
    setForm((prev) => ({
      ...prev,
      line_items: prev.line_items.filter((_, i) => i !== index),
    }))
  }

  async function goToNextReviewItem() {
    try {
      const queue = await getReviewQueue()
      const next = queue.find((item) => item.job_id !== Number(jobId))
      navigate(next ? `/job/${next.job_id}/edit` : '/review')
    } catch {
      navigate('/review')
    }
  }

  async function handleSubmit(event) {
    event.preventDefault()
    setSubmitting(true)
    setActionError(null)
    try {
      await submitJobEdit(jobId, toPayload(form))
      await goToNextReviewItem()
    } catch (err) {
      setActionError(err.message)
      setSubmitting(false)
    }
  }

  async function handleDelete() {
    if (!window.confirm(`Delete job ${jobId}? This cannot be undone.`)) return
    setDeleting(true)
    setActionError(null)
    try {
      await deleteJob(jobId)
      await goToNextReviewItem()
    } catch (err) {
      setActionError(err.message)
      setDeleting(false)
    }
  }

  if (loadError) return <p className="p-10 text-sm text-red-400">{loadError}</p>
  if (form === null) return <p className="p-10 text-sm text-muted">Loading…</p>
  if (form === false) {
    return (
      <div className="p-10">
        <p className="text-sm text-amber-300">This job doesn't exist.</p>
        <Link
          to="/review"
          className="mt-4 inline-block rounded bg-accent px-3 py-1.5 text-sm font-medium text-white hover:bg-accent-hover"
        >
          &larr; Back to Review Queue
        </Link>
      </div>
    )
  }

  return (
    <div className="mx-auto min-h-screen w-full px-[1%] bg-surface">
      <div className="grid grid-cols-2 gap-4 p-2">
        <form
          onSubmit={handleSubmit}
          className="sticky top-10 max-h-[90vh] space-y-2 self-start overflow-y-auto pr-5"
        >
          {reviewInfo?.sourceLabel && (
            <p className="text-xs text-muted">{reviewInfo.sourceLabel}</p>
          )}
          <ReviewIssues reviewInfo={reviewInfo} />
          <div className="grid grid-cols-2 gap-5">
            <div>
              <label className={labelClass}>Document type</label>
              <select
                className={inputClass}
                value={form.document_type}
                onChange={(e) => updateField('document_type', e.target.value)}
              >
                <option value="invoice">Invoice</option>
                <option value="receipt">Receipt</option>
              </select>
            </div>
            <Field
              label="Number"
              value={form.invoice_number}
              onChange={(e) => updateField('invoice_number', e.target.value)}
            />

          </div>
          <div className={`grid gap-5 ${receipt ? 'grid-cols-1' : 'grid-cols-2'}`}>
            <Field
              label="Issue date"
              type="date"
              value={form.issue_date}
              onChange={(e) => updateField('issue_date', e.target.value)}
            />
            {!receipt && (
              <Field
                label="Due date"
                type="date"
                value={form.due_date}
                onChange={(e) => updateField('due_date', e.target.value)}
              />
            )}
          </div>

          <div
            className={`grid gap-5 mt-5 border-t border-edge pt-3 pb-3 ${receipt ? 'grid-cols-1' : 'grid-cols-2'}`}
          >
            <div>
              <h2 className="mb-2 text-sm font-semibold">Vendor</h2>
              <div className="space-y-2">
                <Field
                  label="Name"
                  value={form.vendor.name}
                  onChange={(e) => updateParty('vendor', 'name', e.target.value)}
                />
                <Field
                  label="Address"
                  value={form.vendor.address}
                  onChange={(e) => updateParty('vendor', 'address', e.target.value)}
                />
                <Field
                  label="Tax ID"
                  value={form.vendor.tax_id}
                  onChange={(e) => updateParty('vendor', 'tax_id', e.target.value)}
                />
                <Field
                  label="IBAN"
                  value={form.vendor.iban}
                  onChange={(e) => updateParty('vendor', 'iban', e.target.value)}
                />
              </div>
            </div>
            {!receipt && (
              <div>
                <h2 className="mb-2 text-sm font-semibold">Customer</h2>
                <div className="space-y-2">
                  <Field
                    label="Name"
                    value={form.customer.name}
                    onChange={(e) => updateParty('customer', 'name', e.target.value)}
                  />
                  <Field
                    label="Address"
                    value={form.customer.address}
                    onChange={(e) => updateParty('customer', 'address', e.target.value)}
                  />
                  <Field
                    label="Tax ID"
                    value={form.customer.tax_id}
                    onChange={(e) => updateParty('customer', 'tax_id', e.target.value)}
                  />
                </div>
              </div>
            )}
          </div>
          <div className="border-t border-edge pt-3 pb-3">
            <div className="mb-2 flex items-center justify-between">
              <h2 className="text-sm font-semibold">Line items</h2>
              <button
                type="button"
                onClick={addLineItem}
                className="cursor-pointer rounded border border-edge-strong px-2 py-1 text-xs text-body hover:bg-surface-hover"
              >
                + Add line item
              </button>
            </div>
            <table className="w-full border-collapse text-left text-sm">
              <thead>
                <tr>
                  <th className="py-1 pr-2 font-medium text-muted">#</th>
                  <th className="py-1 pr-2 font-medium text-muted">Description</th>
                  <th className="py-1 pr-2 font-medium text-muted">Unit price</th>
                  <th className="py-1 pr-2 font-medium text-muted">Quantity</th>
                  <th className="py-1 pr-2 font-medium text-muted">Line total</th>
                  <th className="py-1"></th>
                </tr>
              </thead>
              <tbody>
                {form.line_items.map((line, idx) => (
                  <tr
                    key={idx}
                    className={`border-b border-edge ${line.marked ? 'opacity-40' : ''}`}
                  >
                    <td className="py-1 pr-2 text-muted">{idx + 1}</td>
                    <td className={`py-1 pr-2 ${line.marked ? 'line-through' : ''}`}>
                      <input
                        className={inputClass}
                        value={line.description}
                        disabled={line.marked}
                        onChange={(e) => updateLineItem(idx, 'description', e.target.value)}
                      />
                    </td>
                    <td className={`py-1 pr-2 ${line.marked ? 'line-through' : ''}`}>
                      <input
                        className={inputClass}
                        value={line.unit_price}
                        disabled={line.marked}
                        onChange={(e) => updateLineItem(idx, 'unit_price', e.target.value)}
                      />
                    </td>
                    <td className={`py-1 pr-2 ${line.marked ? 'line-through' : ''}`}>
                      <input
                        className={inputClass}
                        value={line.quantity}
                        disabled={line.marked}
                        onChange={(e) => updateLineItem(idx, 'quantity', e.target.value)}
                      />
                    </td>
                    <td className={`py-1 pr-2 ${line.marked ? 'line-through' : ''}`}>
                      <input
                        className={inputClass}
                        value={line.line_total}
                        disabled={line.marked}
                        onChange={(e) => updateLineItem(idx, 'line_total', e.target.value)}
                      />
                    </td>
                    <td className="py-1 text-center">
                      {lineItemHasData(line) ? (
                        <label className="flex cursor-pointer items-center gap-1 text-xs text-red-400">
                          <input
                            type="checkbox"
                            checked={line.marked}
                            onChange={() => toggleLineItemMarked(idx)}
                          />
                          {line.marked ? 'Marked' : 'Remove'}
                        </label>
                      ) : (
                        <button
                          type="button"
                          onClick={() => removeLineItem(idx)}
                          className="cursor-pointer text-xs text-red-400 hover:underline"
                        >
                          Remove
                        </button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="grid grid-cols-5 gap-4 border-t border-edge pt-3">
            <Field
              label="Currency"
              value={form.currency}
              onChange={(e) => updateField('currency', e.target.value)}
            />
            <Field
              label="Subtotal"
              value={form.subtotal}
              onChange={(e) => updateField('subtotal', e.target.value)}
            />
            <Field
              label="Tax"
              value={form.tax}
              onChange={(e) => updateField('tax', e.target.value)}
            />
            <Field
              label="Service charge"
              value={form.service_charge}
              onChange={(e) => updateField('service_charge', e.target.value)}
            />
            <Field
              label="Discount"
              value={form.discount}
              onChange={(e) => updateField('discount', e.target.value)}
            />
          </div>

          <div className="w-48">
            <Field
              label="Grand total"
              value={form.grand_total}
              onChange={(e) => updateField('grand_total', e.target.value)}
              required
            />
          </div>

          {actionError && <p className="text-sm text-red-400">{actionError}</p>}
          <div className='flex items-center gap-3 mt-5'>
            <button
              type="submit"
              disabled={submitting}
              className="cursor-pointer rounded bg-accent px-4 py-2 text-sm font-medium text-white hover:bg-accent-hover disabled:cursor-not-allowed disabled:opacity-40"
            >
              {submitting ? 'Saving…' : 'Save'}
            </button>
            <button
              type="button"
              disabled={submitting}
              onClick={() => navigate('/review')}
              className="cursor-pointer rounded border border-edge-strong px-4 py-2 text-sm font-medium text-body hover:bg-surface-hover disabled:cursor-not-allowed disabled:opacity-40"
            >
              Cancel
            </button>
            <button
              type="button"
              disabled={deleting}
              onClick={handleDelete}
              className="ml-auto cursor-pointer rounded bg-red-600 px-4 py-2 text-sm font-medium text-white hover:bg-red-500 disabled:cursor-not-allowed disabled:opacity-40"
            >
              {deleting ? 'Deleting…' : 'Delete'}
            </button>
          </div>
        </form>
        <div className="flex justify-center">
          <a href={getJobImageUrl(jobId)} target="_blank" rel="noreferrer">
            <img
              src={getJobImageUrl(jobId)}
              alt={`Document for job ${jobId}`}
              className="rounded border border-edge"
            />
          </a>
        </div>
      </div>
    </div>
  )
}
