import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';

export default function ProbabilityChart({ probabilities }) {
  const data = Object.entries(probabilities || {})
    .map(([label, value]) => ({ label, value: Number(value) }))
    .sort((left, right) => right.value - left.value);
  const topClass = data[0]?.label;

  return (
    <section className="card probability-card" aria-labelledby="probability-title">
      <div className="section-heading section-heading--compact">
        <div><span className="eyebrow">CLASS DISTRIBUTION</span><h2 id="probability-title">Model probabilities</h2></div>
        <span className="chart-unit">0–100%</span>
      </div>
      <div role="img" aria-label={`Model probability distribution: ${data.map(({ label, value }) => `${label} ${(value * 100).toFixed(1)} percent`).join(', ')}`} className="probability-chart">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} layout="vertical" margin={{ top: 2, right: 36, left: 2, bottom: 2 }} barSize={13}>
            <CartesianGrid horizontal={false} stroke="#e8eeeb" />
            <XAxis type="number" domain={[0, 1]} tickFormatter={(value) => `${Math.round(value * 100)}%`} tick={{ fill: '#82918a', fontSize: 11 }} axisLine={false} tickLine={false} />
            <YAxis type="category" dataKey="label" width={50} tick={{ fill: '#53655d', fontSize: 12, fontWeight: 600 }} axisLine={false} tickLine={false} />
            <Tooltip formatter={(value) => [`${(Number(value) * 100).toFixed(2)}%`, 'Probability']} cursor={{ fill: '#f4f8f6' }} />
            <Bar dataKey="value" radius={[0, 5, 5, 0]}>
              {data.map((entry) => <Cell key={entry.label} fill={entry.label === topClass ? '#168a6b' : '#b8d9ce'} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <p className="chart-caption">Probabilities describe the model output. They are not a diagnosis or a measure of clinical risk.</p>
    </section>
  );
}

