// Fraudec shared formatters — one definition used by every page.
// Fraudec's dataset is INR-denominated, so all currency display goes
// through this single Intl.NumberFormat('en-IN', ...) instance.
const FRAUDEC_INR = new Intl.NumberFormat('en-IN', {
  style: 'currency',
  currency: 'INR',
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});

function formatMoney(v) {
  return FRAUDEC_INR.format(Number(v) || 0);
}
