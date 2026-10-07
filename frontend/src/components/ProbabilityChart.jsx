import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { getLesionInfo } from '../services/lesionInfo.js';

export default function ProbabilityChart({ probabilities }) {
  const data = Object.entries(probabilities || {})
    .map(([code, value]) => {
      const info = getLesionInfo(code);
      return {
        code,
        label: `${info.shortLabel} (${code})`,
        shortLabel: info.shortLabel,
        fullName: info.friendlyName,
        category: info.category,
        value: Number(value)
      };
    })
    .sort((left, right) => right.value - left.value);

  const topCode = data[0]?.code;

  return (
    <section className="card probability-card" aria-labelledby="probability-title">
      <div className="section-heading section-heading--compact">
        <div>
          <span className="eyebrow">04 / CLASS DISTRIBUTION</span>
          <h2 id="probability-title">Probability distribution across all 7 categories</h2>
        </div>
        <span className="chart-unit">0–100%</span>
      </div>
      <div
        role="img"
        aria-label={`Model probability distribution: ${data.map(({ label, value }) => `${label} ${(value * 100).toFixed(1)} percent`).join(', ')}`}
        className="probability-chart"
      >
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} layout="vertical" margin={{ top: 4, right: 36, left: 12, bottom: 4 }} barSize={15}>
            <CartesianGrid horizontal={false} stroke="#e8eeeb" />
            <XAxis
              type="number"
              domain={[0, 1]}
              tickFormatter={(value) => `${Math.round(value * 100)}%`}
              tick={{ fill: '#82918a', fontSize: 11 }}
              axisLine={false}
              tickLine={false}
            />
            <YAxis
              type="category"
              dataKey="label"
              width={185}
              tick={{ fill: '#41544c', fontSize: 11, fontWeight: 600 }}
              axisLine={false}
              tickLine={false}
            />
            <Tooltip
              formatter={(value, name, item) => [
                `${(Number(value) * 100).toFixed(2)}%`,
                item?.payload?.fullName || 'Probability'
              ]}
              cursor={{ fill: '#f4f8f6' }}
            />
            <Bar dataKey="value" radius={[0, 5, 5, 0]}>
              {data.map((entry) => (
                <Cell key={entry.code} fill={entry.code === topCode ? '#168a6b' : '#b8d9ce'} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <p className="chart-caption">
        Probabilities describe model predictive scores for the 7 HAM10000 diagnostic classes. They do not constitute an official medical diagnosis.
      </p>
    </section>
  );
}
