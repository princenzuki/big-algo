import React from 'react';
import { useNavigate } from 'react-router-dom';

const MetricCard = ({ title, value, change, icon: Icon, subtitle, onClick, clickable = false }) => {
  const navigate = useNavigate();
  const getChangeColor = (change) => {
    switch (change) {
      case 'positive':
        return 'text-success-600';
      case 'negative':
        return 'text-danger-600';
      case 'warning':
        return 'text-warning-600';
      default:
        return 'text-gray-600';
    }
  };

  const getIconColor = (change) => {
    switch (change) {
      case 'positive':
        return 'text-success-500';
      case 'negative':
        return 'text-danger-500';
      case 'warning':
        return 'text-warning-500';
      default:
        return 'text-gray-500';
    }
  };

  const handleClick = () => {
    if (onClick) {
      onClick();
    } else if (clickable) {
      // Default navigation based on title
      switch (title) {
        case 'Account Balance':
        case 'Net Profit':
          navigate('/risk');
          break;
        case 'Total Trades':
        case 'Win Rate':
        case 'Profit Factor':
          navigate('/trades');
          break;
        case 'System Health':
          navigate('/health');
          break;
        default:
          break;
      }
    }
  };

  return (
    <div 
      className={`metric-card ${clickable || onClick ? 'cursor-pointer hover:scale-105 transition-all duration-200 hover:shadow-glow-blue' : ''}`}
      onClick={handleClick}
    >
      <div className="flex items-center">
        <div className="flex-shrink-0">
          {Icon && (
            <Icon className={`h-8 w-8 ${getIconColor(change)}`} />
          )}
        </div>
        <div className="ml-4 flex-1">
          <p className="metric-label">{title}</p>
          <p className={`metric-value ${getChangeColor(change)}`}>
            {value}
          </p>
          {subtitle && (
            <p className="text-xs text-gray-500 mt-1">{subtitle}</p>
          )}
        </div>
      </div>
    </div>
  );
};

export default MetricCard;
