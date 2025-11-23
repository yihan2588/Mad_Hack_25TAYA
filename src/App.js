import React, { useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceArea,
} from "recharts";
import { Upload, Music, TrendingUp, Award, Loader } from "lucide-react";

const ViolonApp = () => {
  const [userFile, setUserFile] = useState(null);
  const [masterFile, setMasterFile] = useState(null);
  const [chartData, setChartData] = useState([]);
  const [accuracyRegions, setAccuracyRegions] = useState([]);
  const [stats, setStats] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const API_URL = "http://localhost:8000";

  const handleUserFileChange = (e) => {
    const file = e.target.files[0];
    if (file) setUserFile(file);
  };

  const handleMasterFileChange = (e) => {
    const file = e.target.files[0];
    if (file) setMasterFile(file);
  };

  const processFiles = async () => {
    if (!userFile || !masterFile) return;

    setIsLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append("student_file", userFile);
    formData.append("master_file", masterFile);

    try {
      const response = await fetch(`${API_URL}/analyze`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Analysis failed. Please try again.");
      }

      const result = await response.json();
      const data = result.chartData || [];
      setChartData(data);
      setStats(result.stats || null);

      const regions = [];
      if (data.length > 0) {
        let currentRegion = {
          start: data[0].time,
          accuracy: data[0].accuracy,
        };

        for (let i = 1; i < data.length; i += 1) {
          if (data[i].accuracy !== currentRegion.accuracy) {
            regions.push({ ...currentRegion, end: data[i - 1].time });
            currentRegion = {
              start: data[i].time,
              accuracy: data[i].accuracy,
            };
          }
        }
        regions.push({ ...currentRegion, end: data[data.length - 1].time });
      }
      setAccuracyRegions(regions);
    } catch (err) {
      console.error(err);
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const getColorForAccuracy = (accuracy) => {
    switch (accuracy) {
      case "good":
        return "rgba(34, 197, 94, 0.15)";
      case "moderate":
        return "rgba(234, 179, 8, 0.15)";
      case "poor":
        return "rgba(239, 68, 68, 0.15)";
      default:
        return "transparent";
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-900 via-purple-800 to-pink-700 p-4 md:p-8">
      <div className="max-w-7xl mx-auto">
        <div className="text-center mb-12 animate-fade-in">
          <div className="inline-block bg-white/10 backdrop-blur-lg rounded-full px-6 py-2 mb-4 border border-white/20">
            <span className="text-white/90 text-sm font-medium">
              🎻 Professional Violin Analysis
            </span>
          </div>
          <h1 className="text-6xl md:text-7xl font-bold text-white mb-4 drop-shadow-2xl">
            Violon
          </h1>
          <p className="text-xl text-purple-100 font-light">
            Master your violin performance with AI-powered analysis
          </p>
        </div>

        <div className="grid md:grid-cols-2 gap-6 mb-8">
          <div className="bg-white/10 backdrop-blur-xl rounded-2xl shadow-2xl p-8 border border-white/20 hover:bg-white/15 transition-all duration-300 transform hover:scale-105">
            <div className="flex items-center mb-6">
              <div className="bg-indigo-500 p-3 rounded-xl mr-3 shadow-lg">
                <Music className="text-white" size={28} />
              </div>
              <h2 className="text-2xl font-bold text-white">Your Performance</h2>
            </div>
            <label className="flex flex-col items-center justify-center border-2 border-dashed border-white/40 rounded-xl p-10 cursor-pointer hover:border-indigo-400 hover:bg-white/5 transition-all duration-300 group">
              <Upload className="text-white/70 mb-3 group-hover:text-indigo-300 transition-colors" size={48} />
              <span className="text-base text-white/80 mb-2 font-medium">Upload Audio or MIDI</span>
              <span className="text-xs text-white/60">.m4a, .mp3, .wav, .mid</span>
              <input type="file" accept="audio/*,.mid,.midi" onChange={handleUserFileChange} className="hidden" />
              {userFile && (
                <div className="mt-4 bg-indigo-500/30 px-4 py-2 rounded-lg border border-indigo-400/50">
                  <span className="text-sm text-white font-medium">✓ {userFile.name}</span>
                </div>
              )}
            </label>
          </div>

          <div className="bg-white/10 backdrop-blur-xl rounded-2xl shadow-2xl p-8 border border-white/20 hover:bg-white/15 transition-all duration-300 transform hover:scale-105">
            <div className="flex items-center mb-6">
              <div className="bg-purple-500 p-3 rounded-xl mr-3 shadow-lg">
                <Award className="text-white" size={28} />
              </div>
              <h2 className="text-2xl font-bold text-white">Master Reference</h2>
            </div>
            <label className="flex flex-col items-center justify-center border-2 border-dashed border-white/40 rounded-xl p-10 cursor-pointer hover:border-purple-400 hover:bg-white/5 transition-all duration-300 group">
              <Upload className="text-white/70 mb-3 group-hover:text-purple-300 transition-colors" size={48} />
              <span className="text-base text-white/80 mb-2 font-medium">Upload Master MIDI/Audio</span>
              <span className="text-xs text-white/60">Reference performance</span>
              <input type="file" accept="audio/*,.mid,.midi" onChange={handleMasterFileChange} className="hidden" />
              {masterFile && (
                <div className="mt-4 bg-purple-500/30 px-4 py-2 rounded-lg border border-purple-400/50">
                  <span className="text-sm text-white font-medium">✓ {masterFile.name}</span>
                </div>
              )}
            </label>
          </div>
        </div>

        <div className="text-center mb-8">
          <button
            onClick={processFiles}
            disabled={!userFile || !masterFile || isLoading}
            className="bg-gradient-to-r from-pink-500 to-violet-600 text-white px-8 py-3 rounded-full font-bold text-lg shadow-lg hover:scale-105 transition-transform disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isLoading ? (
              <span className="flex items-center justify-center">
                <Loader className="animate-spin mr-2" /> Analyzing...
              </span>
            ) : (
              "Analyze Performance"
            )}
          </button>
          {error && (
            <p className="mt-4 text-red-300 bg-red-900/50 p-2 rounded-lg inline-block">{error}</p>
          )}
        </div>

        {stats && !isLoading && (
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8 animate-fade-in">
            <div className="bg-gradient-to-br from-green-500/20 to-green-600/20 backdrop-blur-xl rounded-2xl p-6 border border-green-400/30">
              <div className="flex items-center justify-between mb-2">
                <span className="text-green-100 text-sm font-medium">Overall Score</span>
                <TrendingUp className="text-green-300" size={20} />
              </div>
              <div className="text-4xl font-bold text-white">{stats.overallScore}%</div>
            </div>
            <div className="bg-gradient-to-br from-green-500/20 to-emerald-600/20 backdrop-blur-xl rounded-2xl p-6 border border-green-400/30">
              <span className="text-green-100 text-sm font-medium block mb-2">Excellent</span>
              <div className="text-3xl font-bold text-white">{stats.goodPercent}%</div>
            </div>
            <div className="bg-gradient-to-br from-yellow-500/20 to-amber-600/20 backdrop-blur-xl rounded-2xl p-6 border border-yellow-400/30">
              <span className="text-yellow-100 text-sm font-medium block mb-2">Moderate</span>
              <div className="text-3xl font-bold text-white">{stats.moderatePercent}%</div>
            </div>
            <div className="bg-gradient-to-br from-red-500/20 to-rose-600/20 backdrop-blur-xl rounded-2xl p-6 border border-red-400/30">
              <span className="text-red-100 text-sm font-medium block mb-2">Needs Work</span>
              <div className="text-3xl font-bold text-white">{stats.poorPercent}%</div>
            </div>
          </div>
        )}

        {chartData.length > 0 && !isLoading && (
          <div className="bg-white/10 backdrop-blur-xl rounded-2xl shadow-2xl p-8 border border-white/20 animate-fade-in">
            <div className="flex items-center mb-6">
              <TrendingUp className="text-white mr-3" size={32} />
              <h2 className="text-3xl font-bold text-white">Analysis Graph</h2>
            </div>

            <div className="bg-white rounded-2xl p-6 shadow-xl">
              <ResponsiveContainer width="100%" height={450}>
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis
                    dataKey="time"
                    type="number"
                    domain={["dataMin", "dataMax"]}
                    label={{ value: "Time (s)", position: "insideBottom", offset: -5 }}
                  />
                  <YAxis label={{ value: "Pitch (MIDI)", angle: -90, position: "insideLeft" }} />
                  <Tooltip contentStyle={{ borderRadius: "12px" }} />

                  {accuracyRegions.map((region, idx) => (
                    <ReferenceArea
                      key={`${region.start}-${region.end}-${idx}`}
                      x1={region.start}
                      x2={region.end}
                      fill={getColorForAccuracy(region.accuracy)}
                      fillOpacity={1}
                    />
                  ))}

                  <Line type="monotone" dataKey="master" stroke="#9333ea" strokeWidth={3} name="Master" dot={false} />
                  <Line type="monotone" dataKey="user" stroke="#4f46e5" strokeWidth={3} name="You" dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ViolonApp;
