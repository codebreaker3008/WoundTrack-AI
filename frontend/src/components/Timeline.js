import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Calendar, TrendingDown, Upload } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

const API_BASE_URL = 'http://localhost:8000';

const Timeline = () => {
  const [timelineData, setTimelineData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  useEffect(() => {
    loadTimelineData();
  }, []);

  const loadTimelineData = async () => {
    try {
      setLoading(true);
      
      // Get list of all uploaded wounds from localStorage
      const uploadedWounds = JSON.parse(localStorage.getItem('uploadedWounds') || '[]');
      
      if (uploadedWounds.length === 0) {
        setTimelineData([]);
        setLoading(false);
        return;
      }

      // Fetch details for each wound
      const woundDetails = await Promise.all(
        uploadedWounds.map(async (woundId) => {
          try {
            const response = await axios.get(`${API_BASE_URL}/wound/${woundId}`);
            return response.data;
          } catch (err) {
            console.error(`Error fetching wound ${woundId}:`, err);
            return null;
          }
        })
      );

      // Filter out failed fetches and sort by date (newest first)
      const validWounds = woundDetails
        .filter(w => w !== null)
        .sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));

      setTimelineData(validWounds);
      setError(null);
    } catch (err) {
      console.error('Error loading timeline:', err);
      setError('Failed to load timeline data');
    } finally {
      setLoading(false);
    }
  };

  const handleWoundClick = (woundId) => {
    navigate(`/analysis/${woundId}`);
  };

  const calculateImprovement = (currentArea, previousArea) => {
    if (!previousArea || previousArea === 0) return null;
    return ((previousArea - currentArea) / previousArea * 100).toFixed(1);
  };

  if (loading) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="max-w-6xl mx-auto">
        <div className="bg-red-50 border border-red-200 rounded-lg p-6 text-center">
          <p className="text-red-800">{error}</p>
          <button
            onClick={loadTimelineData}
            className="mt-4 bg-red-600 text-white px-4 py-2 rounded-lg hover:bg-red-700"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto">
      <div className="bg-white rounded-lg shadow-md p-6 mb-6">
        <h2 className="text-2xl font-bold text-gray-800 mb-2 flex items-center">
          <Calendar className="w-6 h-6 mr-2" />
          Wound Healing Timeline
        </h2>
        <p className="text-gray-600">
          Track your wound healing progress over time
        </p>
      </div>

      {timelineData.length === 0 ? (
        <div className="bg-white rounded-lg shadow-md p-12 text-center">
          <Calendar className="w-16 h-16 text-gray-400 mx-auto mb-4" />
          <h3 className="text-xl font-semibold text-gray-800 mb-2">
            No Photos Yet
          </h3>
          <p className="text-gray-600 mb-6">
            Start tracking your wound healing by uploading your first photo
          </p>
          <button
            onClick={() => navigate('/upload')}
            className="bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 font-medium inline-flex items-center"
          >
            <Upload className="w-5 h-5 mr-2" />
            Upload First Photo
          </button>
        </div>
      ) : (
        <>
          {/* Summary Stats */}
          <div className="grid md:grid-cols-3 gap-4 mb-6">
            <div className="bg-white rounded-lg shadow-md p-4">
              <p className="text-sm text-gray-600 mb-1">Total Photos</p>
              <p className="text-3xl font-bold text-blue-600">{timelineData.length}</p>
            </div>
            <div className="bg-white rounded-lg shadow-md p-4">
              <p className="text-sm text-gray-600 mb-1">Latest Area</p>
              <p className="text-3xl font-bold text-green-600">
                {timelineData[0]?.measurements?.area_cm2 || 0} cm²
              </p>
            </div>
            <div className="bg-white rounded-lg shadow-md p-4">
              <p className="text-sm text-gray-600 mb-1">Average Risk</p>
              <p className="text-3xl font-bold text-yellow-600">
                {timelineData.length > 0 
                  ? (timelineData.reduce((sum, w) => sum + (w.infection?.probability || 0), 0) / timelineData.length * 100).toFixed(0)
                  : 0}%
              </p>
            </div>
          </div>

          {/* Timeline Grid */}
          <div className="grid md:grid-cols-3 gap-6">
            {timelineData.map((entry, index) => {
              const improvement = index < timelineData.length - 1 
                ? calculateImprovement(
                    entry.measurements?.area_cm2,
                    timelineData[index + 1]?.measurements?.area_cm2
                  )
                : null;

              return (
                <div
                  key={entry.wound_id}
                  className="bg-white rounded-lg shadow-md overflow-hidden hover:shadow-lg transition-shadow cursor-pointer"
                  onClick={() => handleWoundClick(entry.wound_id)}
                >
                  {/* Image Preview */}
                  <div className="bg-gradient-to-br from-blue-500 to-blue-600 h-48 flex items-center justify-center text-white relative">
                    <img
                      src={`${API_BASE_URL}${entry.files?.overlay || '/results/placeholder.jpg'}`}
                      alt="Wound"
                      className="w-full h-full object-cover"
                      onError={(e) => {
                        e.target.style.display = 'none';
                        e.target.parentElement.innerHTML = '<div class="text-center"><svg class="w-12 h-12 mx-auto mb-2 opacity-75" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"></path></svg><p class="text-sm">Wound Photo</p></div>';
                      }}
                    />
                    {index === 0 && (
                      <div className="absolute top-2 right-2 bg-green-500 text-white px-2 py-1 rounded text-xs font-bold">
                        Latest
                      </div>
                    )}
                  </div>
                  
                  <div className="p-4">
                    <div className="flex justify-between items-start mb-3">
                      <div>
                        <p className="text-sm text-gray-600">Date</p>
                        <p className="font-semibold text-gray-800">
                          {new Date(entry.timestamp).toLocaleDateString('en-US', {
                            month: 'short',
                            day: 'numeric',
                            year: 'numeric'
                          })}
                        </p>
                        <p className="text-xs text-gray-500">
                          {new Date(entry.timestamp).toLocaleTimeString('en-US', {
                            hour: '2-digit',
                            minute: '2-digit'
                          })}
                        </p>
                      </div>
                      <div className="text-right">
                        <p className="text-sm text-gray-600">Area</p>
                        <p className="font-semibold text-blue-600">
                          {entry.measurements?.area_cm2 || 0} cm²
                        </p>
                      </div>
                    </div>
                    
                    <div className="flex items-center justify-between pt-3 border-t border-gray-200">
                      <span className="text-sm text-gray-600">Infection Risk</span>
                      <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                        entry.infection?.level === 'Low' ? 'bg-green-100 text-green-800' :
                        entry.infection?.level === 'Medium' ? 'bg-yellow-100 text-yellow-800' :
                        'bg-red-100 text-red-800'
                      }`}>
                        {entry.infection?.level || 'Unknown'}
                      </span>
                    </div>

                    {improvement !== null && (
                      <div className={`mt-3 pt-3 border-t border-gray-200 flex items-center ${
                        parseFloat(improvement) > 0 ? 'text-green-600' : 'text-red-600'
                      }`}>
                        <TrendingDown className="w-4 h-4 mr-1" />
                        <span className="text-sm font-medium">
                          {improvement > 0 ? '+' : ''}{improvement}% from previous
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>

          {/* Actions */}
          <div className="mt-6 flex justify-center">
            <button
              onClick={() => navigate('/upload')}
              className="bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 font-medium inline-flex items-center"
            >
              <Upload className="w-5 h-5 mr-2" />
              Upload New Photo
            </button>
          </div>
        </>
      )}
    </div>
  );
};

export default Timeline;