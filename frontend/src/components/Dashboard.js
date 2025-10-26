import React from 'react';
import { Link } from 'react-router-dom';
import { Camera, TrendingUp, Calendar, FileText } from 'lucide-react';

const Dashboard = () => {
  // Mock data - in real app, fetch from API
  const stats = {
    totalPhotos: 12,
    daysTracking: 21,
    healingProgress: 35,
    lastUpload: '2 days ago'
  };

  const recentWounds = [
    {
      id: '1',
      date: '2024-10-23',
      area: 4.5,
      infectionRisk: 'Low',
      progress: '+15%'
    },
    {
      id: '2',
      date: '2024-10-20',
      area: 5.2,
      infectionRisk: 'Low',
      progress: '+10%'
    },
    {
      id: '3',
      date: '2024-10-17',
      area: 5.8,
      infectionRisk: 'Medium',
      progress: '+8%'
    }
  ];

  return (
    <div className="space-y-6">
      {/* Welcome Section */}
      <div className="bg-gradient-to-r from-blue-600 to-blue-700 rounded-lg shadow-lg p-8 text-white">
        <h1 className="text-3xl font-bold mb-2">Welcome to Wound Healing Tracker</h1>
        <p className="text-blue-100 mb-6">
          Track your wound healing progress with AI-powered analysis
        </p>
        <Link
          to="/upload"
          className="inline-flex items-center bg-white text-blue-600 px-6 py-3 rounded-lg font-semibold hover:bg-blue-50 transition-colors"
        >
          <Camera className="w-5 h-5 mr-2" />
          Upload New Photo
        </Link>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-white rounded-lg shadow-md p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 mb-1">Total Photos</p>
              <p className="text-3xl font-bold text-gray-800">{stats.totalPhotos}</p>
            </div>
            <Camera className="w-10 h-10 text-blue-600 opacity-75" />
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-md p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 mb-1">Days Tracking</p>
              <p className="text-3xl font-bold text-gray-800">{stats.daysTracking}</p>
            </div>
            <Calendar className="w-10 h-10 text-green-600 opacity-75" />
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-md p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 mb-1">Healing Progress</p>
              <p className="text-3xl font-bold text-green-600">+{stats.healingProgress}%</p>
            </div>
            <TrendingUp className="w-10 h-10 text-green-600 opacity-75" />
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-md p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 mb-1">Last Upload</p>
              <p className="text-xl font-semibold text-gray-800">{stats.lastUpload}</p>
            </div>
            <FileText className="w-10 h-10 text-purple-600 opacity-75" />
          </div>
        </div>
      </div>

      {/* Recent Activity */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-xl font-bold text-gray-800">Recent Wound Assessments</h2>
          <Link
            to="/timeline"
            className="text-blue-600 hover:text-blue-700 text-sm font-medium"
          >
            View All →
          </Link>
        </div>

        <div className="space-y-4">
          {recentWounds.map((wound) => (
            <div
              key={wound.id}
              className="flex items-center justify-between p-4 border border-gray-200 rounded-lg hover:border-blue-300 transition-colors cursor-pointer"
              onClick={() => window.location.href = `/analysis/${wound.id}`}
            >
              <div className="flex items-center space-x-4">
                <div className="bg-blue-100 rounded-lg p-3">
                  <Camera className="w-6 h-6 text-blue-600" />
                </div>
                <div>
                  <p className="font-semibold text-gray-800">
                    {new Date(wound.date).toLocaleDateString('en-US', {
                      month: 'long',
                      day: 'numeric',
                      year: 'numeric'
                    })}
                  </p>
                  <p className="text-sm text-gray-600">
                    Area: {wound.area} cm² • Infection Risk: {' '}
                    <span className={`font-medium ${
                      wound.infectionRisk === 'Low' ? 'text-green-600' :
                      wound.infectionRisk === 'Medium' ? 'text-yellow-600' :
                      'text-red-600'
                    }`}>
                      {wound.infectionRisk}
                    </span>
                  </p>
                </div>
              </div>
              <div className="text-right">
                <p className="text-lg font-bold text-green-600">{wound.progress}</p>
                <p className="text-sm text-gray-500">healing</p>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Quick Actions */}
      <div className="grid md:grid-cols-3 gap-4">
        <Link
          to="/upload"
          className="bg-white rounded-lg shadow-md p-6 hover:shadow-lg transition-shadow"
        >
          <Camera className="w-8 h-8 text-blue-600 mb-3" />
          <h3 className="font-semibold text-gray-800 mb-2">Take New Photo</h3>
          <p className="text-sm text-gray-600">
            Upload a new wound photo for AI analysis
          </p>
        </Link>

        <Link
          to="/compare"
          className="bg-white rounded-lg shadow-md p-6 hover:shadow-lg transition-shadow"
        >
          <TrendingUp className="w-8 h-8 text-green-600 mb-3" />
          <h3 className="font-semibold text-gray-800 mb-2">Compare Progress</h3>
          <p className="text-sm text-gray-600">
            Compare two photos to see healing progress
          </p>
        </Link>

        <Link
          to="/timeline"
          className="bg-white rounded-lg shadow-md p-6 hover:shadow-lg transition-shadow"
        >
          <Calendar className="w-8 h-8 text-purple-600 mb-3" />
          <h3 className="font-semibold text-gray-800 mb-2">View Timeline</h3>
          <p className="text-sm text-gray-600">
            See all your wound photos in chronological order
          </p>
        </Link>
      </div>

      {/* Tips Section */}
      <div className="bg-gradient-to-r from-green-50 to-blue-50 border border-green-200 rounded-lg p-6">
        <h3 className="font-semibold text-gray-800 mb-3">💡 Pro Tips for Better Tracking</h3>
        <ul className="space-y-2 text-sm text-gray-700">
          <li className="flex items-start">
            <span className="text-green-600 mr-2">✓</span>
            Take photos at the same time of day for consistent lighting
          </li>
          <li className="flex items-start">
            <span className="text-green-600 mr-2">✓</span>
            Include a reference object (coin/ruler) for accurate measurements
          </li>
          <li className="flex items-start">
            <span className="text-green-600 mr-2">✓</span>
            Upload photos every 3-4 days to track healing velocity
          </li>
          <li className="flex items-start">
            <span className="text-green-600 mr-2">✓</span>
            Consult your healthcare provider if infection risk increases
          </li>
        </ul>
      </div>
    </div>
  );
};

export default Dashboard;