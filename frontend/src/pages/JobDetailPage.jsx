import { useEffect, useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import { getJobDetail, getJobImageUrl } from '../lib/api'
import ReviewIssues from '../components/ReviewIssues'
import { isReceipt, startingInvoice, toReviewInfo } from '../lib/invoiceForm'

const labelClass = 'block text-xs font-medium text-muted mb-1'

function displayValue(value) {
  if (value === null || value === undefined || value === '') return '—'
  return String(value)
}

function ReadOnlyField({ label, value }) {
  return (
    <div>
      <label className={labelClass}>{label}</label>
      <p className="text-sm">{displayValue(value)}</p>
    </div>
  )
}

const DOCUMENT_TYPE_LABELS = {
  invoice: 'Invoice',
  receipt: 'Receipt',
}

export default function JobDetailPage() {
  const { jobId } = useParams()
  const [form, setForm] = useState(null)
  const [error, setError] = useState(null)
  const [reviewInfo, setReviewInfo] = useState(null)

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
          setError(err.message)
        }
      }
    }
    load()
  }, [jobId])

  if (error) return <p className="p-10 text-sm text-red-400">{error}</p>
  if (form === null) return <p className="p-10 text-sm text-muted">Loading…</p>
  if (form === false) {
    return (
      <div className="p-10">
        <p className="text-sm text-amber-300">This job doesn't exist.</p>
        <Link
          to="/ledger"
          className="mt-4 inline-block rounded bg-accent px-3 py-1.5 text-sm font-medium text-white hover:bg-accent-hover"
        >
          &larr; Back to Ledger
        </Link>
      </div>
    )
  }

  const receipt = isReceipt(form.document_type)

  return (
    <div className="mx-auto min-h-screen w-full px-[1%] bg-surface">
      <div className="grid grid-cols-2 gap-4 p-2">
        <div className="sticky top-10 max-h-[90vh] space-y-2 self-start overflow-y-auto pr-5">
          {reviewInfo?.sourceLabel && (
            <p className="text-xs text-muted">{reviewInfo.sourceLabel}</p>
          )}
          <ReviewIssues reviewInfo={reviewInfo} />
          <div className="grid grid-cols-2 gap-5">
            <div>
              <label className={labelClass}>Document type</label>
              <p className="text-sm">
                {DOCUMENT_TYPE_LABELS[form.document_type] || form.document_type}
              </p>
            </div>
            <ReadOnlyField label="Number" value={form.invoice_number} />
          </div>
          <div className={`grid gap-5 ${receipt ? 'grid-cols-1' : 'grid-cols-2'}`}>
            <ReadOnlyField label="Issue date" value={form.issue_date} />
            {!receipt && <ReadOnlyField label="Due date" value={form.due_date} />}
          </div>

          <div
            className={`grid gap-5 mt-5 border-t border-edge pt-3 pb-3 ${receipt ? 'grid-cols-1' : 'grid-cols-2'}`}
          >
            <div>
              <h2 className="mb-2 text-sm font-semibold">Vendor</h2>
              <div className="space-y-2">
                <ReadOnlyField label="Name" value={form.vendor.name} />
                <ReadOnlyField label="Address" value={form.vendor.address} />
                <ReadOnlyField label="Tax ID" value={form.vendor.tax_id} />
                <ReadOnlyField label="IBAN" value={form.vendor.iban} />
              </div>
            </div>
            {!receipt && (
              <div>
                <h2 className="mb-2 text-sm font-semibold">Customer</h2>
                <div className="space-y-2">
                  <ReadOnlyField label="Name" value={form.customer.name} />
                  <ReadOnlyField label="Address" value={form.customer.address} />
                  <ReadOnlyField label="Tax ID" value={form.customer.tax_id} />
                </div>
              </div>
            )}
          </div>

          <div className="border-t border-edge pt-3 pb-3">
            <h2 className="mb-2 text-sm font-semibold">Line items</h2>
            <table className="w-full border-collapse text-left text-sm">
              <thead>
                <tr className="border-b border-edge">
                  <th className="py-1 pr-2 font-medium text-muted">#</th>
                  <th className="py-1 pr-2 font-medium text-muted">Description</th>
                  <th className="py-1 pr-2 font-medium text-muted">Unit price</th>
                  <th className="py-1 pr-2 font-medium text-muted">Quantity</th>
                  <th className="py-1 pr-2 font-medium text-muted">Line total</th>
                </tr>
              </thead>
              <tbody>
                {form.line_items.map((line, idx) => (
                  <tr key={idx} className="border-b border-edge">
                    <td className="py-1 pr-2 text-muted">{idx + 1}</td>
                    <td className="py-1 pr-2">{displayValue(line.description)}</td>
                    <td className="py-1 pr-2">{displayValue(line.unit_price)}</td>
                    <td className="py-1 pr-2">{displayValue(line.quantity)}</td>
                    <td className="py-1 pr-2">{displayValue(line.line_total)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="grid grid-cols-5 gap-4 pt-3">
            <ReadOnlyField label="Currency" value={form.currency} />
            <ReadOnlyField label="Subtotal" value={form.subtotal} />
            <ReadOnlyField label="Tax" value={form.tax} />
            <ReadOnlyField label="Service charge" value={form.service_charge} />
            <ReadOnlyField label="Discount" value={form.discount} />
          </div>

          <div className="w-48">
            <ReadOnlyField label="Grand total" value={form.grand_total} />
          </div>

          <div className="flex items-center gap-3 mt-5">
            <Link
              to={`/job/${jobId}/edit`}
              className="rounded bg-accent px-4 py-2 text-sm font-medium text-white hover:bg-accent-hover"
            >
              Edit
            </Link>
            <Link
              to="/ledger"
              className="rounded border border-edge-strong px-4 py-2 text-sm font-medium text-body hover:bg-surface-hover"
            >
              Back
            </Link>
          </div>
        </div>

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
