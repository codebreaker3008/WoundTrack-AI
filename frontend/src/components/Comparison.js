import React, { useState } from 'react';
import axios from 'axios';
import { ArrowRight, TrendingDown, Calendar } from 'lucide-react';

const API_BASE_URL = 'http://localhost:8000';

const Comparison = () => {
  const [wound1Id, setWound1Id] = useState('');
  const [wound2Id, setWound2Id] = useState('');
  const [comparisonResult, setComparisonResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleCompare = async () => {
    if (!wound1Id || !wound2Id) {
      setError('Please enter both wound IDs');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await axios.post(`${API_BASE_URL}/compare-wounds`, {
        wound_id_1: wound1Id,
        wound_id_2: wound2Id
      });

      setComparisonResult(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to compare wounds');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto">
      <div className="bg-white rounded-lg shadow-md p-6 mb-6">
        <h2 className="text-2xl font-bold text-gray-800 mb-2">
          Compare Wound Progress
        </h2>
        <p className="text-gray-600">
          Compare two wound photos to see healing progress
        </p>
      </div>

      {/* Input Section */}
      <div className="bg-white rounded-lg shadow-md p-6 mb-6">
        <div className="grid md:grid-cols-3 gap-4 items-end">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Earlier Wound ID
            </label>
            <input
              type="text"
              value={wound1Id}
              onChange={(e) => setWound1Id(e.target.value)}
              placeholder="Enter first wound ID"
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
          </div>

          <div className="flex justify-center">
            <ArrowRight className="w-6 h-6 text-gray-400" />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Later Wound ID
            </label>
            <input
              type="text"
              value={wound2Id}
              onChange={(e) => setWound2Id(e.target.value)}
              placeholder="Enter second wound ID"
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
          </div>
        </div>

        {error && (
          <div className="mt-4 bg-red-50 border border-red-200 rounded-lg p-3 text-red-800 text-sm">
            {error}
          </div>
        )}

        <button
          onClick={handleCompare}
          disabled={loading}
          className="mt-4 w-full bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed font-medium"
        >
          {loading ? 'Comparing...' : 'Compare Wounds'}
        </button>
      </div>

      {/* Results Section */}
      {comparisonResult && (
        <div className="space-y-6">
          {/* Healing Progress Card */}
          <div className="bg-gradient-to-br from-green-50 to-blue-50 rounded-lg shadow-md p-6 border border-green-200">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-xl font-bold text-gray-800">
                Healing Progress
              </h3>
              <TrendingDown className="w-8 h-8 text-green-600" />
            </div>

            <div className="grid md:grid-cols-2 gap-6">
              <div className="text-center">
                <p className="text-sm text-gray-600 mb-1">Healing Percentage</p>
                <p className="text-4xl font-bold text-green-600">
                  {comparisonResult.healing_percentage > 0 ? '+' : ''}
                  {comparisonResult.healing_percentage.toFixed(1)}%
                </p>
              </div>

              <div className="text-center">
                <p className="text-sm text-gray-600 mb-1">Area Reduction</p>
                <p className="text-4xl font-bold text-blue-600">
                  {comparisonResult.area_reduction_cm2.toFixed(2)} cm²
                </p>
              </div>
            </div>
          </div>

          {/* Metrics Grid */}
          <div className="grid md:grid-cols-2 gap-6">
            <div className="bg-white rounded-lg shadow-md p-6">
              <div className="flex items-center mb-2">
                <Calendar className="w-5 h-5 text-blue-600 mr-2" />
                <h4 className="font-semibold text-gray-800">Days Between</h4>
              </div>
              <p className="text-3xl font-bold text-gray-800">
                {comparisonResult.days_between}
              </p>
              <p className="text-sm text-gray-600 mt-1">days</p>
            </div>

            <div className="bg-white rounded-lg shadow-md p-6">
              <div className="flex items-center mb-2">
                <TrendingDown className="w-5 h-5 text-green-600 mr-2" />
                <h4 className="font-semibold text-gray-800">Healing Velocity</h4>
              </div>
              <p className="text-3xl font-bold text-gray-800">
                {comparisonResult.healing_velocity.toFixed(3)}
              </p>
              <p className="text-sm text-gray-600 mt-1">cm²/day</p>
            </div>
          </div>

          {/* Recommendation */}
          <div className={`rounded-lg shadow-md p-6 ${
            comparisonResult.healing_percentage > 20
              ? 'bg-green-50 border border-green-200'
              : comparisonResult.healing_percentage > 10
              ? 'bg-blue-50 border border-blue-200'
              : 'bg-yellow-50 border border-yellow-200'
          }`}>
            <h4 className="font-semibold text-gray-800 mb-2">Recommendation</h4>
            <p className={`${
              comparisonResult.healing_percentage > 20
                ? 'text-green-800'
                : comparisonResult.healing_percentage > 10
                ? 'text-blue-800'
                : 'text-yellow-800'
            }`}>
              {comparisonResult.recommendation}
            </p>
          </div>
        </div>
      )}

      {/* Instructions */}
      {!comparisonResult && (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-6">
          <h3 className="font-semibold text-blue-900 mb-3">How to Compare:</h3>
          <ol className="list-decimal list-inside text-sm text-blue-800 space-y-2">
            <li>Upload wound photos from different dates</li>
            <li>Note the wound IDs from the analysis results</li>
            <li>Enter the earlier wound ID in the first field</li>
            <li>Enter the later wound ID in the second field</li>
            <li>Click "Compare Wounds" to see healing progress</li>
          </ol>
        </div>
      )}
    </div>
  );
};

export default Comparison;