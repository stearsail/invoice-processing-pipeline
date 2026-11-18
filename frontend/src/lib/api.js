const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8001'

async function request(path, options) {
  const response = await fetch(`${API_BASE}${path}`, options)
  if (!response.ok) {
    const text = await response.text()
    let message = text
    try {
      // FastAPI errors are {"detail": "..."}; show the message, not raw JSON
      const detail = JSON.parse(text).detail
      if (typeof detail === 'string') message = detail
    } catch {
      // not JSON: keep the raw body
    }
    const error = new Error(`${response.status}: ${message}`)
    error.status = response.status
    throw error
  }
  return response.json()
}

export function uploadImages(files) {
  const formData = new FormData()
  for (const file of files) {
    formData.append('files', file)
  }
  return request('/extraction/upload_image', { method: 'POST', body: formData })
}

export function getJobStatus(jobId) {
  return request(`/extraction/status/${jobId}`)
}

export function getReviewQueue() {
  return request('/ledger/review')
}

export function getResolvedLedger() {
  return request('/ledger/resolved')
}

export function getPendingJobs() {
  return request('/ledger/pending')
}

export function getErroredJobs() {
  return request('/ledger/errored')
}

export function getJobDetail(jobId) {
  return request(`/ledger/${jobId}`)
}

export function submitJobEdit(jobId, invoice) {
  return request(`/ledger/${jobId}/edit`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(invoice),
  })
}

export function getJobImageUrl(jobId) {
  return `${API_BASE}/extraction/image/${jobId}`
}

export function deleteJob(jobId) {
  return request(`/ledger/${jobId}`, { method: 'DELETE' })
}

export function retryJob(jobId) {
  return request(`/ledger/${jobId}/retry`, { method: 'POST' })
}