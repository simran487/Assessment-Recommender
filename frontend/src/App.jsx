import React, { useState, useEffect } from 'react';
import { Search, Loader2, Link as LinkIcon, Briefcase, Zap, AlertTriangle, BrainCircuit } from 'lucide-react';

// Configuration for the FastAPI endpoint
const API_BASE_URL = (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1')
    ? 'http://127.0.0.1:8000' // Local development
    : ''; // Deployed environment (relative path)

const API_ENDPOINT = `${API_BASE_URL}/recommend`;

// Define the structure for a single recommendation (must match the FastAPI Pydantic model)
const initialRecommendations = [
  { Assessment_name: "Python Coding Challenge", Assessment_url: "#", Test_type: ["Knowledge & Skills"] },
  { Assessment_name: "Interpersonal Communications", Assessment_url: "#", Test_type: ["Competencies", "Personality & Behaviour"] },
];

/**
 * Custom hook to implement exponential backoff for API calls
 * @param {function} apiCall - The async function to execute.
 * @param {number} maxRetries - Maximum number of retries.
 * @returns {Promise<any>} The result of the successful API call.
 */
const useFetchWithBackoff = () => {
  const fetchWithBackoff = async (apiCall, maxRetries = 5, delay = 1000) => {
    for (let i = 0; i < maxRetries; i++) {
      try {
        const response = await apiCall();
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        return await response.json();
      } catch (error) {
        console.error(`Attempt ${i + 1} failed:`, error);
        if (i === maxRetries - 1) {
          throw new Error("API failed after maximum retries.");
        }
        // Wait using exponential backoff (delay * 2^i)
        const currentDelay = delay * Math.pow(2, i);
        await new Promise(resolve => setTimeout(resolve, currentDelay));
      }
    }
  };
  return fetchWithBackoff;
};

// Main App Component
const App = () => {
  const fetchWithBackoff = useFetchWithBackoff();
  const [query, setQuery] = useState('');
  const [recommendations, setRecommendations] = useState(initialRecommendations);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Function to call the FastAPI recommendation endpoint
  const getRecommendations = async () => {
    if (!query.trim()) {
      setError("Please enter a job description to get recommendations.");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const apiCall = () => fetch(API_ENDPOINT, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ user_query: query }),
      });

      const data = await fetchWithBackoff(apiCall);
      setRecommendations(data);

    } catch (err) {
      console.error("Failed to fetch recommendations:", err);
      setRecommendations(initialRecommendations); // Revert to initial on error
      setError(`Recommendation failed. Ensure the FastAPI service is running at ${API_BASE_URL}. Details: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    getRecommendations();
  };

  // Component to render individual assessment cards
  const RecommendationCard = ({ rec, index }) => {
    const typeIcons = {
      'Knowledge & Skills': <Briefcase className="w-4 h-4 text-sky-600" />,
      'Competencies': <Zap className="w-4 h-4 text-emerald-600" />,
      'Personality & Behaviour': <Zap className="w-4 h-4 text-amber-600" />,
      'Cognitive': <BrainCircuit className="w-4 h-4 text-fuchsia-600" />,
    };

    return (
      <div className="bg-white p-6 shadow-lg rounded-xl border border-gray-100 hover:shadow-xl transition duration-300">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-xl font-bold text-gray-800 leading-snug">
            {index + 1}. {rec.Assessment_name}
          </h3>
          <span className="text-sm font-semibold text-gray-500">
            Rank {index + 1}
          </span>
        </div>
        
        <div className="flex flex-wrap gap-2 mb-4">
          {rec.Test_type && rec.Test_type.map(type => (
            <span key={type} className="flex items-center space-x-1 px-3 py-1 bg-gray-50 text-xs font-medium rounded-full text-gray-600 border border-gray-200">
              {typeIcons[type] || <Briefcase className="w-4 h-4 text-gray-500" />}
              <span>{type}</span>
            </span>
          ))}
        </div>

        <a 
          href={rec.Assessment_url || '#'} 
          target="_blank" 
          rel="noopener noreferrer" 
          className="inline-flex items-center text-indigo-600 hover:text-indigo-800 transition duration-150 font-medium"
        >
          <LinkIcon className="w-4 h-4 mr-1" />
          View Assessment URL
        </a>
      </div>
    );
  };


  return (
    <div className="min-h-screen bg-gray-50 p-4 sm:p-8 font-sans">
      <header className="text-center mb-10">
        <h1 className="text-4xl font-extrabold text-gray-900 mb-2">
          Assessment Recommender (RAG)
        </h1>
        <p className="text-lg text-gray-600">
          Enter a job description to find the most relevant SHL assessments.
        </p>
      </header>

      {/* Input Form */}
      <div className="max-w-3xl mx-auto mb-12 bg-white p-6 rounded-xl shadow-2xl border-t-4 border-indigo-500">
        <form onSubmit={handleSubmit} className="space-y-4">
          <label htmlFor="query" className="block text-sm font-medium text-gray-700">
            Job Description Query:
          </label>
          <textarea
            id="query"
            rows="4"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="e.g., 'Looking for a Senior Data Scientist proficient in Python, SQL, and predictive modeling, assessment duration max 60 min.'"
            className="w-full border-2 border-gray-300 rounded-lg p-3 focus:ring-indigo-500 focus:border-indigo-500 transition duration-150 shadow-inner"
            required
            disabled={loading}
          />
          <button
            type="submit"
            className="w-full inline-flex justify-center items-center px-6 py-3 border border-transparent text-base font-medium rounded-lg shadow-sm text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 transition duration-150 disabled:opacity-50"
            disabled={loading}
          >
            {loading ? (
              <>
                <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                Searching...
              </>
            ) : (
              <>
                <Search className="w-5 h-5 mr-2" />
                Find Recommendations
              </>
            )}
          </button>
        </form>
      </div>

      {/* Status & Results */}
      <div className="max-w-5xl mx-auto">
        {error && (
          <div className="bg-red-100 border-l-4 border-red-500 text-red-700 p-4 rounded-lg mb-8" role="alert">
            <div className="flex items-center">
              <AlertTriangle className="w-5 h-5 mr-2" />
              <p className="font-bold">Error</p>
            </div>
            <p className="text-sm mt-1">{error}</p>
          </div>
        )}

        {recommendations.length > 0 && !loading && !error && (
          <div className="space-y-6">
            <h2 className="text-2xl font-bold text-gray-800 border-b pb-2 mb-4">
              Top {recommendations.length} Recommended Assessments
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {recommendations.map((rec, index) => (
                <RecommendationCard key={rec.Assessment_url + index} rec={rec} index={index} />
              ))}
            </div>
          </div>
        )}

        {/* Initial state message */}
        {recommendations.length === 0 && !loading && !error && !query && (
            <div className="text-center p-12 bg-white rounded-xl shadow-lg">
                <p className="text-gray-500">Enter a query above to see the recommendations appear here.</p>
            </div>
        )}
      </div>

    </div>
  );
};

export default App;