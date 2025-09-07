import React, { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { 
  HomeIcon, 
  ChartBarIcon, 
  ShieldCheckIcon, 
  HeartIcon, 
  CogIcon,
  CpuChipIcon,
  Bars3Icon,
  XMarkIcon
} from '@heroicons/react/24/outline';
import { useApi } from '../contexts/ApiContext';

const Layout = ({ children }) => {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const location = useLocation();
  const { dashboardData, lastUpdate } = useApi();

  const navigation = [
    { name: 'Dashboard', href: '/', icon: HomeIcon },
    { name: 'Trades', href: '/trades', icon: ChartBarIcon },
    { name: 'AI Insights', href: '/ai-insights', icon: CpuChipIcon },
    { name: 'Risk', href: '/risk', icon: ShieldCheckIcon },
    { name: 'Health', href: '/health', icon: HeartIcon },
    { name: 'Settings', href: '/settings', icon: CogIcon },
  ];

  const isCurrentPath = (path) => {
    if (path === '/') {
      return location.pathname === '/';
    }
    return location.pathname.startsWith(path);
  };

  const formatLastUpdate = (date) => {
    if (!date) return 'Never';
    const now = new Date();
    const diff = now - date;
    const minutes = Math.floor(diff / 60000);
    
    if (minutes < 1) return 'Just now';
    if (minutes < 60) return `${minutes}m ago`;
    const hours = Math.floor(minutes / 60);
    if (hours < 24) return `${hours}h ago`;
    const days = Math.floor(hours / 24);
    return `${days}d ago`;
  };

  return (
    <div className="min-h-screen bg-bg-primary">
      {/* Mobile sidebar */}
      <div className={`fixed inset-0 z-50 lg:hidden ${sidebarOpen ? 'block' : 'hidden'}`}>
        <div className="fixed inset-0 bg-black bg-opacity-50 backdrop-blur-sm" onClick={() => setSidebarOpen(false)} />
        <div className="fixed inset-y-0 left-0 flex w-64 flex-col bg-bg-secondary border-r border-dark-700">
          <div className="flex h-16 items-center justify-between px-4 border-b border-dark-700">
            <h1 className="text-xl font-bold text-gradient">BigAlgo FinTech</h1>
            <button
              onClick={() => setSidebarOpen(false)}
              className="text-text-muted hover:text-text-primary transition-colors"
            >
              <XMarkIcon className="h-6 w-6" />
            </button>
          </div>
          <nav className="flex-1 space-y-1 px-2 py-4">
            {navigation.map((item) => (
              <Link
                key={item.name}
                to={item.href}
                onClick={() => setSidebarOpen(false)}
                className={`group flex items-center px-3 py-3 text-sm font-medium rounded-xl transition-all duration-300 ${
                  isCurrentPath(item.href)
                    ? 'bg-primary-500/20 text-primary-400 border border-primary-500/30'
                    : 'text-text-muted hover:bg-dark-700/50 hover:text-text-primary'
                }`}
              >
                <item.icon
                  className={`mr-3 h-5 w-5 transition-colors ${
                    isCurrentPath(item.href)
                      ? 'text-primary-400'
                      : 'text-text-muted group-hover:text-text-primary'
                  }`}
                />
                {item.name}
              </Link>
            ))}
          </nav>
          
          {/* Status footer */}
          <div className="p-4 border-t border-dark-700">
            <div className="text-xs text-text-muted">
              <div className="flex items-center justify-between mb-2">
                <span>Last update:</span>
                <span className="font-medium text-text-secondary">{formatLastUpdate(lastUpdate)}</span>
              </div>
              {dashboardData?.system_status && (
                <div className="flex items-center justify-between">
                  <span>Status:</span>
                  <div className="flex items-center">
                    <div className={`w-2 h-2 rounded-full mr-2 ${
                      dashboardData.system_status.broker_connected 
                        ? 'bg-success-500 animate-pulse' 
                        : 'bg-danger-500'
                    }`} />
                    <span className="font-medium text-text-secondary">
                      {dashboardData.system_status.broker_connected ? 'Online' : 'Offline'}
                    </span>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Desktop sidebar */}
      <div className="hidden lg:fixed lg:inset-y-0 lg:flex lg:w-64 lg:flex-col">
        <div className="flex flex-col flex-grow bg-bg-secondary border-r border-dark-700 backdrop-blur-xl">
          <div className="flex h-16 items-center px-4 border-b border-dark-700">
            <h1 className="text-xl font-bold text-gradient">BigAlgo FinTech</h1>
          </div>
          <nav className="flex-1 space-y-1 px-2 py-4">
            {navigation.map((item) => (
              <Link
                key={item.name}
                to={item.href}
                className={`group flex items-center px-3 py-3 text-sm font-medium rounded-xl transition-all duration-300 ${
                  isCurrentPath(item.href)
                    ? 'bg-primary-500/20 text-primary-400 border border-primary-500/30 shadow-glow-blue'
                    : 'text-text-muted hover:bg-dark-700/50 hover:text-text-primary'
                }`}
              >
                <item.icon
                  className={`mr-3 h-5 w-5 transition-colors ${
                    isCurrentPath(item.href)
                      ? 'text-primary-400'
                      : 'text-text-muted group-hover:text-text-primary'
                  }`}
                />
                {item.name}
              </Link>
            ))}
          </nav>
          
          {/* Status footer */}
          <div className="p-4 border-t border-dark-700">
            <div className="text-xs text-text-muted">
              <div className="flex items-center justify-between mb-2">
                <span>Last update:</span>
                <span className="font-medium text-text-secondary">{formatLastUpdate(lastUpdate)}</span>
              </div>
              {dashboardData?.system_status && (
                <div className="flex items-center justify-between">
                  <span>Status:</span>
                  <div className="flex items-center">
                    <div className={`w-2 h-2 rounded-full mr-2 ${
                      dashboardData.system_status.broker_connected 
                        ? 'bg-success-500 animate-pulse' 
                        : 'bg-danger-500'
                    }`} />
                    <span className="font-medium text-text-secondary">
                      {dashboardData.system_status.broker_connected ? 'Online' : 'Offline'}
                    </span>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Main content */}
      <div className="lg:pl-64">
        {/* Top bar */}
        <div className="sticky top-0 z-40 flex h-16 shrink-0 items-center gap-x-4 border-b border-dark-700 bg-bg-secondary/80 backdrop-blur-xl px-4 shadow-glass sm:gap-x-6 sm:px-6 lg:px-8">
          <button
            type="button"
            className="-m-2.5 p-2.5 text-text-muted hover:text-text-primary lg:hidden transition-colors"
            onClick={() => setSidebarOpen(true)}
          >
            <Bars3Icon className="h-6 w-6" />
          </button>

          <div className="flex flex-1 gap-x-4 self-stretch lg:gap-x-6">
            <div className="flex flex-1" />
            <div className="flex items-center gap-x-4 lg:gap-x-6">
              {/* Health indicator */}
              {dashboardData?.algo_health && (
                <div className="flex items-center gap-x-2 px-3 py-1 rounded-lg bg-dark-800/50 border border-dark-600">
                  <div className={`w-2 h-2 rounded-full ${
                    dashboardData.algo_health.status === 'healthy' 
                      ? 'bg-success-500 animate-pulse' 
                      : dashboardData.algo_health.status === 'warning'
                      ? 'bg-warning-500'
                      : 'bg-danger-500'
                  }`} />
                  <span className="text-sm font-medium text-text-secondary">
                    Health: {dashboardData.algo_health.health_score}/100
                  </span>
                </div>
              )}
              
              {/* Risk indicator */}
              {dashboardData?.risk_summary && (
                <div className="flex items-center gap-x-2 px-3 py-1 rounded-lg bg-dark-800/50 border border-dark-600">
                  <div className="text-sm text-text-muted">Risk:</div>
                  <div className="text-sm font-medium text-text-secondary">
                    {dashboardData.risk_summary.current_risk_percent.toFixed(1)}%
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Page content */}
        <main className="py-6">
          <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
            {children}
          </div>
        </main>
      </div>
    </div>
  );
};

export default Layout;
