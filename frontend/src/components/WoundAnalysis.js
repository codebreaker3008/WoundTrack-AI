import React, { useState, useEffect } from 'react';
import { useParams, useLocation } from 'react-router-dom';
import axios from 'axios';
import { AlertTriangle, CheckCircle, Info, TrendingDown, Activity } from 'lucide-react';

const API_BASE_URL = 'http://localhost:8000';

const WoundAnalysis = () => {
  const { woundId } = useParams();
  const location = useLocation();
  const [data, setData] = useState(location.state?.analysisData || null);
  const [loading, setLoading] = useState(!data);

  useEffect(() => {
    if (!data) {
      fetchWoundData();
    }
  }, [woundId]);

  const fetchWoundData = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/wound/${woundId}`);
      setData(response.data);
    } catch (error) {
      console.error('Error fetching wound data:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="text-center py-12">
        <p className="text-gray-600">Unable to load wound data</p>
      </div>
    );
  }

  // Get infection risk color
  const getRiskColor = (level) => {
    switch (level.toLowerCase()) {
      case 'high':
        return 'red';
      case 'medium':
        return 'yellow';
      case 'low':
        return 'green';
      default:
        return 'gray';
    }
  };

  const riskColor = getRiskColor(data.infection_level || 'Low');

  return (
    <div className="max-w-6xl mx-auto">
      {/* Header */}
      <div className="bg-white rounded-lg shadow-md p-6 mb-6">
        <h2 className="text-2xl font-bold text-gray-800 mb-2">
          Wound Analysis Results
        </h2>
        <p className="text-gray-600">
          Analysis Date: {new Date(data.timestamp).toLocaleString()}
        </p>
        {data.patient_id && (
          <p className="text-gray-600">Patient ID: {data.patient_id}</p>
        )}
      </div>

      <div className="grid md:grid-cols-2 gap-6">
        {/* Left Column - Images */}
        <div className="space-y-6">
          {/* Segmentation Visualization */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <h3 className="text-lg font-semibold text-gray-800 mb-4">
              Wound Segmentation
            </h3>
            <img
              src={`${API_BASE_URL}${data.segmentation_mask}`}
              alt="Wound Segmentation"
              className="w-full rounded-lg"
              onError={(e) => {
                e.target.src = 'https://via.placeholder.com/500x400?text=Image+Not+Available';
              }}
            />
            <p className="text-sm text-gray-600 mt-2">
              Red overlay indicates the detected wound area
            </p>
          </div>

          {/* Infection Risk Gauge */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <h3 className="text-lg font-semibold text-gray-800 mb-4">
              Infection Risk Assessment
            </h3>
            
            {/* Risk Level Badge */}
            <div className="flex items-center justify-center mb-4">
              <div className={`
                px-6 py-3 rounded-full text-xl font-bold
                ${riskColor === 'red' ? 'bg-red-100 text-red-800' : ''}
                ${riskColor === 'yellow' ? 'bg-yellow-100 text-yellow-800' : ''}
                ${riskColor === 'green' ? 'bg-green-100 text-green-800' : ''}
              `}>
                {data.infection_level || 'Low'} Risk
              </div>
            </div>

            {/* Risk Percentage */}
            <div className="mb-4">
              <div className="flex justify-between text-sm text-gray-600 mb-1">
                <span>Infection Probability</span>
                <span className="font-semibold">
                  {((data.infection_risk || 0) * 100).toFixed(1)}%
                </span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-4">
                <div
                  className={`h-4 rounded-full transition-all duration-500 ${
                    riskColor === 'red' ? 'bg-red-600' :
                    riskColor === 'yellow' ? 'bg-yellow-500' :
                    'bg-green-500'
                  }`}
                  style={{ width: `${(data.infection_risk || 0) * 100}%` }}
                ></div>
              </div>
            </div>

            {/* Risk Interpretation */}
            <div className={`
              rounded-lg p-4 flex items-start
              ${riskColor === 'red' ? 'bg-red-50 border border-red-200' : ''}
              ${riskColor === 'yellow' ? 'bg-yellow-50 border border-yellow-200' : ''}
              ${riskColor === 'green' ? 'bg-green-50 border border-green-200' : ''}
            `}>
              {riskColor === 'red' && <AlertTriangle className="w-5 h-5 text-red-600 mt-0.5 mr-2 flex-shrink-0" />}
              {riskColor === 'yellow' && <Info className="w-5 h-5 text-yellow-600 mt-0.5 mr-2 flex-shrink-0" />}
              {riskColor === 'green' && <CheckCircle className="w-5 h-5 text-green-600 mt-0.5 mr-2 flex-shrink-0" />}
              <div>
                <h4 className={`font-semibold mb-1 ${
                  riskColor === 'red' ? 'text-red-900' :
                  riskColor === 'yellow' ? 'text-yellow-900' :
                  'text-green-900'
                }`}>
                  {riskColor === 'red' && 'Seek Medical Attention'}
                  {riskColor === 'yellow' && 'Monitor Closely'}
                  {riskColor === 'green' && 'Continue Current Care'}
                </h4>
                <p className={`text-sm ${
                  riskColor === 'red' ? 'text-red-800' :
                  riskColor === 'yellow' ? 'text-yellow-800' :
                  'text-green-800'
                }`}>
                  {riskColor === 'red' && 'High infection risk detected. Consult healthcare provider promptly.'}
                  {riskColor === 'yellow' && 'Moderate infection signs. Watch for worsening symptoms.'}
                  {riskColor === 'green' && 'Low infection risk. Wound appears to be healing normally.'}
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Right Column - Measurements */}
        <div className="space-y-6">
          {/* Wound Measurements */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center">
              <Activity className="w-5 h-5 mr-2" />
              Wound Measurements
            </h3>
            
            <div className="space-y-4">
              {/* Area */}
              <div className="border-b border-gray-200 pb-3">
                <div className="flex justify-between items-center">
                  <span className="text-gray-600">Area</span>
                  <span className="text-2xl font-bold text-blue-600">
                    {data.area_cm2 || 0} cm²
                  </span>
                </div>
              </div>

              {/* Length */}
              <div className="border-b border-gray-200 pb-3">
                <div className="flex justify-between items-center">
                  <span className="text-gray-600">Length</span>
                  <span className="text-xl font-semibold text-gray-800">
                    {data.length_cm || 0} cm
                  </span>
                </div>
              </div>

              {/* Width */}
              <div className="border-b border-gray-200 pb-3">
                <div className="flex justify-between items-center">
                  <span className="text-gray-600">Width</span>
                  <span className="text-xl font-semibold text-gray-800">
                    {data.width_cm || 0} cm
                  </span>
                </div>
              </div>

              {/* Perimeter */}
              <div className="pb-3">
                <div className="flex justify-between items-center">
                  <span className="text-gray-600">Perimeter</span>
                  <span className="text-xl font-semibold text-gray-800">
                    {data.perimeter_cm || 0} cm
                  </span>
                </div>
              </div>
            </div>

            <div className="mt-4 bg-blue-50 border border-blue-200 rounded-lg p-3">
              <p className="text-sm text-blue-800">
                <strong>Note:</strong> Measurements are estimates. For most accurate results, 
                include a reference object (coin/ruler) in your photos.
              </p>
            </div>
          </div>

          {/* Recommendations */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <h3 className="text-lg font-semibold text-gray-800 mb-4">
              Care Recommendations
            </h3>
            
            <div className="space-y-3">
              <div className="flex items-start">
                <CheckCircle className="w-5 h-5 text-green-600 mt-0.5 mr-2 flex-shrink-0" />
                <p className="text-sm text-gray-700">
                  Keep wound clean and dry
                </p>
              </div>
              <div className="flex items-start">
                <CheckCircle className="w-5 h-5 text-green-600 mt-0.5 mr-2 flex-shrink-0" />
                <p className="text-sm text-gray-700">
                  Change dressings as recommended by healthcare provider
                </p>
              </div>
              <div className="flex items-start">
                <CheckCircle className="w-5 h-5 text-green-600 mt-0.5 mr-2 flex-shrink-0" />
                <p className="text-sm text-gray-700">
                  Monitor for signs of infection (increased redness, swelling, discharge)
                </p>
              </div>
              <div className="flex items-start">
                <CheckCircle className="w-5 h-5 text-green-600 mt-0.5 mr-2 flex-shrink-0" />
                <p className="text-sm text-gray-700">
                  Take regular photos to track healing progress
                </p>
              </div>
            </div>

            {riskColor === 'red' && (
              <div className="mt-4 bg-red-50 border border-red-200 rounded-lg p-3">
                <p className="text-sm text-red-800 font-semibold">
                  ⚠️ Seek professional medical advice promptly due to high infection risk.
                </p>
              </div>
            )}
          </div>

          {/* Actions */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <h3 className="text-lg font-semibold text-gray-800 mb-4">
              Next Steps
            </h3>
            <div className="space-y-3">
              <button 
                onClick={async () => {
                  try {
                    const response = await axios.get(
                      `${API_BASE_URL}/wound/${woundId}/report`,
                      { responseType: 'blob' }
                    );
                    const url = window.URL.createObjectURL(new Blob([response.data]));
                    const link = document.createElement('a');
                    link.href = url;
                    link.setAttribute('download', `wound_report_${woundId}.pdf`);
                    document.body.appendChild(link);
                    link.click();
                    link.remove();
                  } catch (error) {
                    alert('Failed to download report. Please try again.');
                    console.error('Report download error:', error);
                  }
                }}
                className="w-full bg-blue-600 text-white px-4 py-3 rounded-lg hover:bg-blue-700 font-medium"
              >
                Download Report (PDF)
              </button>
              <button 
                onClick={() => window.location.href = '/compare'}
                className="w-full border border-blue-600 text-blue-600 px-4 py-3 rounded-lg hover:bg-blue-50 font-medium"
              >
                Compare with Previous Photo
              </button>
              <button 
                onClick={() => window.location.href = '/timeline'}
                className="w-full border border-gray-300 text-gray-700 px-4 py-3 rounded-lg hover:bg-gray-50 font-medium"
              >
                View Timeline
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Disclaimer */}
      <div className="mt-6 bg-yellow-50 border border-yellow-200 rounded-lg p-4">
        <p className="text-sm text-yellow-800">
          <strong>Medical Disclaimer:</strong> This tool provides AI-assisted wound analysis 
          for informational purposes only. It is not a substitute for professional medical 
          advice, diagnosis, or treatment. Always consult with qualified healthcare 
          professionals for wound care decisions.
        </p>
      </div>
    </div>
  );
};

export default WoundAnalysis;