import React from 'react';

const MetricCard = ({ title, value, change, icon: Icon, subtitle }) => {
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

  return (
    <div className="metric-card">
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
