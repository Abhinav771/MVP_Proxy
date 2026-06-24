import React, { useEffect, useState } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement
} from 'chart.js';
import { Bar, Doughnut } from 'react-chartjs-2';
import { Activity, Database, Server, DollarSign, AlertCircle, CreditCard } from 'lucide-react';
import './App.css';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement
);

function App() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeChart, setActiveChart] = useState(null);

  useEffect(() => {
    fetchData();
    // Refresh every 5 seconds
    const interval = setInterval(fetchData, 5000);
    return () => clearInterval(interval);
  }, []);

  const fetchData = async () => {
    try {
      const response = await fetch('http://127.0.0.1:8000/admin/api/dashboard');
      if (!response.ok) throw new Error('Network response was not ok');
      const result = await response.json();
      
      // Also fetch users
      const usersResponse = await fetch('http://127.0.0.1:8000/admin/users');
      let usersResult = { users: [] };
      if (usersResponse.ok) {
        usersResult = await usersResponse.json();
      }
      
      setData({ ...result, users: usersResult.users });
      setLoading(false);
    } catch (err) {
      setError(err.message);
      setLoading(false);
    }
  };

  if (loading) return <div className="loading">Loading dashboard...</div>;
  if (error) return <div className="error">Error loading dashboard: {error}</div>;
  if (!data) return null;

  // Chart configs
  const cacheData = {
    labels: ['Cache Hits', 'Cache Misses'],
    datasets: [{
      data: [data.cache_hits, data.cache_misses],
      backgroundColor: ['#10b981', '#334155'],
      borderColor: ['#059669', '#1e293b'],
      borderWidth: 1,
    }]
  };

  const modelData = {
    labels: ['Small Model', 'Large Model'],
    datasets: [{
      data: [data.route_small, data.route_large],
      backgroundColor: ['#6366f1', '#f59e0b'],
      borderColor: ['#4f46e5', '#d97706'],
      borderWidth: 1,
    }]
  };
  
  const hitRate = data.cache_hits + data.cache_misses > 0 
    ? ((data.cache_hits / (data.cache_hits + data.cache_misses)) * 100).toFixed(1) 
    : 0;

  const smallModelRate = data.route_small + data.route_large > 0
    ? ((data.route_small / (data.route_small + data.route_large)) * 100).toFixed(1)
    : 0;

  return (
    <div className="dashboard">
      <header className="header">
        <h1>MVP Proxy Dashboard</h1>
        <div className="date-badge">{data.date}</div>
      </header>

      <div className="hero-grid">
        <div 
          className={`card stat-card clickable ${activeChart === 'requests' ? 'active' : ''}`}
          onClick={() => setActiveChart(activeChart === 'requests' ? null : 'requests')}
        >
          <div className="stat-icon bg-blue"><Activity size={24} /></div>
          <div>
            <h3>Total Requests</h3>
            <div className="stat-value">{data.total_requests}</div>
          </div>
        </div>
        <div 
          className={`card stat-card clickable ${activeChart === 'financials' ? 'active' : ''}`}
          onClick={() => setActiveChart(activeChart === 'financials' ? null : 'financials')}
        >
          <div className="stat-icon bg-orange"><CreditCard size={24} /></div>
          <div>
            <h3>Total Actual Cost</h3>
            <div className="stat-value">${(data.actual_cost || 0).toFixed(4)}</div>
          </div>
        </div>
        <div className="card stat-card">
          <div className="stat-icon bg-green"><DollarSign size={24} /></div>
          <div>
            <h3>Estimated Saved</h3>
            <div className="stat-value">${data.estimated_savings.toFixed(4)}</div>
          </div>
        </div>
        <div className="card stat-card">
          <div className="stat-icon bg-emerald"><Database size={24} /></div>
          <div>
            <h3>Cache Hit Rate</h3>
            <div className="stat-value">{hitRate}%</div>
          </div>
        </div>
        <div className="card stat-card">
          <div className="stat-icon bg-purple"><Server size={24} /></div>
          <div>
            <h3>Small Model Usage</h3>
            <div className="stat-value">{smallModelRate}%</div>
          </div>
        </div>
        
      </div>

      {activeChart === 'requests' && data.history && (
        <div className="historical-chart">
          <h3>Historical Requests (7 Days)</h3>
          <div className="chart-container" style={{ height: '300px' }}>
            <Bar 
              data={{
                labels: data.history.dates,
                datasets: [{
                  label: 'Total Requests',
                  data: data.history.requests,
                  backgroundColor: 'rgba(59, 130, 246, 0.8)',
                }]
              }}
              options={{ maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { y: { beginAtZero: true, grid: { color: '#334155', drawTicks: false } }, x: { grid: { color: '#334155', drawTicks: false } } } }}
            />
          </div>
        </div>
      )}

      {activeChart === 'financials' && data.history && (
        <div className="historical-chart">
          <h3>Financial Overview (Cost vs Savings)</h3>
          <div className="chart-container" style={{ height: '300px' }}>
            <Bar 
              data={{
                labels: data.history.dates,
                datasets: [
                  {
                    label: 'Actual Cost ($)',
                    data: data.history.actual_cost,
                    backgroundColor: 'rgba(245, 158, 11, 0.8)',
                  },
                  {
                    label: 'Estimated Savings ($)',
                    data: data.history.estimated_savings,
                    backgroundColor: 'rgba(16, 185, 129, 0.8)',
                  }
                ]
              }}
              options={{ 
                maintainAspectRatio: false, 
                scales: { 
                  x: { stacked: true, grid: { color: '#334155', drawTicks: false } }, 
                  y: { stacked: true, beginAtZero: true, grid: { color: '#334155', drawTicks: false } } 
                },
                plugins: { legend: { position: 'bottom', labels: { color: '#f8fafc' } } }
              }}
            />
          </div>
        </div>
      )}

      <div className="main-grid">
        <div className="card chart-card">
          <h3>Semantic Caching</h3>
          <div className="chart-container">
            <Doughnut data={cacheData} options={{ maintainAspectRatio: false, plugins: { legend: { position: 'bottom', labels: { color: '#f8fafc' } } } }} />
          </div>
        </div>

        <div className="card chart-card">
          <h3>Routing (Small vs Large)</h3>
          <div className="chart-container">
            <Doughnut data={modelData} options={{ maintainAspectRatio: false, plugins: { legend: { position: 'bottom', labels: { color: '#f8fafc' } } } }} />
          </div>
        </div>

        <div className="card list-card">
          <h3>Top Cached Prompts</h3>
          <ul className="prompt-list">
            {data.top_cached_prompts.length === 0 && <li className="empty">No cached prompts yet</li>}
            {data.top_cached_prompts.map((p, i) => (
              <li key={i}>
                <span className="prompt-text">"{p.prompt}"</span>
                <span className="badge">{p.count} hits</span>
              </li>
            ))}
          </ul>
        </div>
      </div>

      <div className="main-grid">
         <div className="card list-card wide-card">
          <h3>Recent Volatile Prompts</h3>
          <ul className="prompt-list">
            {data.recent_volatile.length === 0 && <li className="empty">No volatile requests yet</li>}
            {data.recent_volatile.map((p, i) => (
              <li key={i} className="volatile-item">
                <AlertCircle size={16} className="text-orange" />
                <span className="prompt-text">"{p}"</span>
              </li>
            ))}
          </ul>
        </div>
        
        <div className="card list-card wide-card">
          <h3>Active Users Token Budget</h3>
          <div className="table-container">
            <table>
              <thead>
                <tr>
                  <th>IP Address</th>
                  <th>Small Model Usage</th>
                  <th>Large Model Usage</th>
                  <th>Actual Cost</th>
                </tr>
              </thead>
              <tbody>
                {data.users.length === 0 && (
                  <tr><td colSpan="3" className="empty">No active users today</td></tr>
                )}
                {data.users.map((u, i) => {
                  const smallPct = (u.small_model.used / u.small_model.limit) * 100;
                  const largePct = (u.large_model.used / u.large_model.limit) * 100;
                  return (
                    <tr key={i}>
                      <td className="ip-cell">{u.ip}</td>
                      <td>
                        <div className="progress-bg">
                          <div className={`progress-bar ${smallPct > 80 ? 'danger' : ''}`} style={{ width: `${Math.min(smallPct, 100)}%` }}></div>
                        </div>
                        <div className="progress-text">{u.small_model.used.toLocaleString()} / {u.small_model.limit.toLocaleString()}</div>
                      </td>
                      <td>
                        <div className="progress-bg">
                          <div className={`progress-bar ${largePct > 80 ? 'danger' : ''}`} style={{ width: `${Math.min(largePct, 100)}%` }}></div>
                        </div>
                        <div className="progress-text">{u.large_model.used.toLocaleString()} / {u.large_model.limit.toLocaleString()}</div>
                      </td>
                      <td>
                        <div className="progress-text" style={{ textAlign: 'left', fontSize: '0.875rem', fontWeight: 500 }}>
                          ${(u.actual_cost || 0).toFixed(4)}
                        </div>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
