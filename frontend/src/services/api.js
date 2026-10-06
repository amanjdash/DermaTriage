const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000').replace(/\/$/, '');

async function request(path, options = {}) {
  let response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, options);
  } catch {
    throw new Error('The DermaTriage API could not be reached. Check that the backend is running.');
  }
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = payload?.detail;
    throw new Error(typeof detail === 'string' ? detail : detail?.message || `Request failed (${response.status}).`);
  }
  return payload;
}

export function getHealth() {
  return request('/health');
}

export function analyzeImage(file) {
  const body = new FormData();
  body.append('image', file);
  return request('/predict', { method: 'POST', body });
}

