export const PROFILE_IDS = Array.from({ length: 10 }, (_, index) => `user_${String(index + 1).padStart(2, '0')}`)

const PROFILE_DISPLAY_NAMES = {
  user_01: 'Student',
  user_02: 'Doctor',
  user_03: 'Software Engineer',
  user_04: 'Business Owner',
  user_05: 'Teacher',
  user_06: 'Freelancer',
  user_07: 'Retail Professional',
  user_08: 'Consultant',
  user_09: 'Researcher',
  user_10: 'Finance Professional',
}

export function getProfileDisplayName(profileId) {
  return PROFILE_DISPLAY_NAMES[profileId] || 'No dataset selected'
}
