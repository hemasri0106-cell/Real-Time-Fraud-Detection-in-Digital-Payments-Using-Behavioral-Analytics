const DEVICE_ID_KEY = 'fraudec_device_id'

function generateDeviceId() {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) {
    return `dev_${crypto.randomUUID()}`
  }
  return `dev_${Date.now().toString(36)}${Math.random().toString(36).slice(2, 10)}`
}

export function getDeviceId() {
  let id = localStorage.getItem(DEVICE_ID_KEY)
  if (!id) {
    id = generateDeviceId()
    localStorage.setItem(DEVICE_ID_KEY, id)
  }
  return id
}
