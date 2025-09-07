import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import Trades from './pages/Trades';
import AIInsights from './pages/AIInsights';
import Risk from './pages/Risk';
import Health from './pages/Health';
import Settings from './pages/Settings';
import { ApiProvider } from './contexts/ApiContext';

function App() {
  return (
    <ApiProvider>
      <Router>
        <div className="App">
          <Layout>
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/trades" element={<Trades />} />
              <Route path="/ai-insights" element={<AIInsights />} />
              <Route path="/risk" element={<Risk />} />
              <Route path="/health" element={<Health />} />
              <Route path="/settings" element={<Settings />} />
            </Routes>
          </Layout>
          <Toaster
            position="top-right"
            toastOptions={{
              duration: 4000,
              style: {
                background: '#363636',
                color: '#fff',
              },
              success: {
                duration: 3000,
                iconTheme: {
                  primary: '#22c55e',
                  secondary: '#fff',
                },
              },
              error: {
                duration: 5000,
                iconTheme: {
                  primary: '#ef4444',
                  secondary: '#fff',
                },
              },
            }}
          />
        </div>
      </Router>
    </ApiProvider>
  );
}

export default App;
