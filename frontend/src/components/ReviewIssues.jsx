import { CATEGORY_STYLES, friendlyJobError } from '../lib/invoiceForm'

export default function ReviewIssues({ reviewInfo }) {
  if (!reviewInfo) return null
  const { jobError, issues, loadErrors } = reviewInfo
  if (!jobError && issues.length === 0 && loadErrors.length === 0) return null

  return (
    <div className="mb-6 rounded border border-amber-500/30 bg-amber-500/10 px-4 py-2">
      <h2 className="text-sm font-semibold text-amber-300">Issues:</h2>
      {loadErrors.length > 0 && (
        <p className="text-sm text-red-300" title={loadErrors.join('\n')}>
          Some stored data for this job couldn't be loaded, so parts of the form may be
          blank.
        </p>
      )}
      {jobError && (
        <p className="text-sm text-red-300" title={jobError}>
          {friendlyJobError(jobError)}
        </p>
      )}
      {issues.length > 0 && (
        <ul className="list-disc space-y-1 pl-5 text-sm">
          {issues.map((issue, idx) => (
            <li key={idx} className={CATEGORY_STYLES[issue.category] || 'text-body'}>
              <span className="text-xs font-medium uppercase opacity-70">
                {issue.category}
              </span>{' '}
              — {issue.message}
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
