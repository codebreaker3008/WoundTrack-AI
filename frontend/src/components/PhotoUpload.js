import React, { useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import { Camera, Upload, AlertCircle, CheckCircle } from 'lucide-react';

const API_BASE_URL = 'http://localhost:8000';

const PhotoUpload = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [qualityCheck, setQualityCheck] = useState(null);
  const [patientId, setPatientId] = useState('');
  const fileInputRef = useRef(null);
  const navigate = useNavigate();

  // Handle file selection
  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      processFile(file);
    }
  };

  // Process selected file
  const processFile = (file) => {
    // Validate file type
    if (!file.type.startsWith('image/')) {
      alert('Please select an image file');
      return;
    }

    // Validate file size (max 5MB)
    if (file.size > 5 * 1024 * 1024) {
      alert('File size must be less than 5MB');
      return;
    }

    setSelectedFile(file);

    // Create preview
    const reader = new FileReader();
    reader.onloadend = () => {
      setPreview(reader.result);
      checkImageQuality(reader.result);
    };
    reader.readAsDataURL(file);
  };

  // Basic image quality check
  const checkImageQuality = (imageData) => {
    const img = new Image();
    img.onload = () => {
      const checks = {
        resolution: img.width >= 640 && img.height >= 480,
        aspectRatio: img.width / img.height > 0.5 && img.width / img.height < 2,
      };

      const allPassed = Object.values(checks).every(check => check);
      
      setQualityCheck({
        passed: allPassed,
        checks: {
          'Resolution': checks.resolution ? 'Good' : 'Too low (min 640x480)',
          'Aspect Ratio': checks.aspectRatio ? 'Good' : 'Image too narrow/wide',
        }
      });
    };
    img.src = imageData;
  };

  // Handle upload
  const handleUpload = async () => {
    if (!selectedFile) {
      alert('Please select a file first');
      return;
    }

    if (qualityCheck && !qualityCheck.passed) {
      const confirm = window.confirm('Image quality checks failed. Continue anyway?');
      if (!confirm) return;
    }

    setUploading(true);

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);
      if (patientId) {
        formData.append('patient_id', patientId);
      }

      const response = await axios.post(`${API_BASE_URL}/upload-wound`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      // Save wound ID to localStorage for timeline
      const uploadedWounds = JSON.parse(localStorage.getItem('uploadedWounds') || '[]');
      uploadedWounds.push(response.data.wound_id);
      localStorage.setItem('uploadedWounds', JSON.stringify(uploadedWounds));

      // Navigate to analysis page
      navigate(`/analysis/${response.data.wound_id}`, { 
        state: { analysisData: response.data } 
      });

    } catch (error) {
      console.error('Upload error:', error);
      alert('Failed to upload image. Please try again.');
    } finally {
      setUploading(false);
    }
  };

  // Reset form
  const handleReset = () => {
    setSelectedFile(null);
    setPreview(null);
    setQualityCheck(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="max-w-4xl mx-auto">
      <div className="bg-white rounded-lg shadow-md p-6">
        <h2 className="text-2xl font-bold text-gray-800 mb-6">
          Upload Wound Photo
        </h2>

        {/* Instructions */}
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-6">
          <h3 className="font-semibold text-blue-900 mb-2">Photo Guidelines:</h3>
          <ul className="list-disc list-inside text-sm text-blue-800 space-y-1">
            <li>Position wound in center of frame</li>
            <li>Ensure adequate lighting (not too dark or bright)</li>
            <li>Include a reference object (coin/ruler) for scale if possible</li>
            <li>Take photo from 6-12 inches away</li>
            <li>Avoid blurry images - hold camera steady</li>
          </ul>
        </div>

        {/* Patient ID Input */}
        <div className="mb-6">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Patient ID (Optional)
          </label>
          <input
            type="text"
            value={patientId}
            onChange={(e) => setPatientId(e.target.value)}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            placeholder="Enter patient identifier"
          />
        </div>

        {/* Upload Area */}
        <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center mb-6">
          {!preview ? (
            <div>
              <Camera className="mx-auto h-12 w-12 text-gray-400 mb-4" />
              <p className="text-gray-600 mb-4">
                Take a photo or upload from your device
              </p>
              <div className="flex justify-center space-x-4">
                <button
                  onClick={() => fileInputRef.current.click()}
                  className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 flex items-center"
                >
                  <Upload className="w-5 h-5 mr-2" />
                  Choose File
                </button>
              </div>
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={handleFileSelect}
                className="hidden"
              />
            </div>
          ) : (
            <div>
              <img
                src={preview}
                alt="Preview"
                className="max-w-full max-h-96 mx-auto rounded-lg mb-4"
              />
              <button
                onClick={handleReset}
                className="text-blue-600 hover:text-blue-800 text-sm"
              >
                Choose Different Image
              </button>
            </div>
          )}
        </div>

        {/* Quality Check Results */}
        {qualityCheck && (
          <div className={`rounded-lg p-4 mb-6 ${
            qualityCheck.passed 
              ? 'bg-green-50 border border-green-200' 
              : 'bg-yellow-50 border border-yellow-200'
          }`}>
            <div className="flex items-start">
              {qualityCheck.passed ? (
                <CheckCircle className="h-5 w-5 text-green-600 mt-0.5 mr-2" />
              ) : (
                <AlertCircle className="h-5 w-5 text-yellow-600 mt-0.5 mr-2" />
              )}
              <div>
                <h4 className={`font-semibold ${
                  qualityCheck.passed ? 'text-green-900' : 'text-yellow-900'
                }`}>
                  {qualityCheck.passed ? 'Image Quality: Good' : 'Image Quality: Needs Review'}
                </h4>
                <ul className="mt-2 text-sm space-y-1">
                  {Object.entries(qualityCheck.checks).map(([key, value]) => (
                    <li key={key} className={
                      qualityCheck.passed ? 'text-green-800' : 'text-yellow-800'
                    }>
                      <strong>{key}:</strong> {value}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        )}

        {/* Upload Button */}
        <div className="flex justify-end space-x-4">
          <button
            onClick={handleReset}
            disabled={!selectedFile || uploading}
            className="px-6 py-2 border border-gray-300 rounded-lg text-gray-700 hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Cancel
          </button>
          <button
            onClick={handleUpload}
            disabled={!selectedFile || uploading}
            className="bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center"
          >
            {uploading ? (
              <>
                <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2"></div>
                Analyzing...
              </>
            ) : (
              'Analyze Wound'
            )}
          </button>
        </div>

        {/* Processing Info */}
        {uploading && (
          <div className="mt-6 bg-blue-50 border border-blue-200 rounded-lg p-4">
            <div className="flex items-center">
              <div className="animate-pulse flex space-x-2">
                <div className="w-2 h-2 bg-blue-600 rounded-full"></div>
                <div className="w-2 h-2 bg-blue-600 rounded-full animation-delay-200"></div>
                <div className="w-2 h-2 bg-blue-600 rounded-full animation-delay-400"></div>
              </div>
              <p className="ml-3 text-sm text-blue-800">
                Processing image with AI models... This may take a few moments.
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Tips Section */}
      <div className="mt-6 bg-white rounded-lg shadow-md p-6">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">
          Tips for Best Results
        </h3>
        <div className="grid md:grid-cols-2 gap-4 text-sm text-gray-600">
          <div>
            <h4 className="font-medium text-gray-800 mb-2">📸 Photography</h4>
            <ul className="space-y-1 list-disc list-inside">
              <li>Use natural daylight when possible</li>
              <li>Avoid shadows on the wound</li>
              <li>Keep camera parallel to wound surface</li>
            </ul>
          </div>
          <div>
            <h4 className="font-medium text-gray-800 mb-2">📏 Scale Reference</h4>
            <ul className="space-y-1 list-disc list-inside">
              <li>Place a coin or ruler near the wound</li>
              <li>Keep reference object at same level as wound</li>
              <li>This helps calculate accurate measurements</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PhotoUpload;