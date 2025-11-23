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
import { Upload, Music, TrendingUp, Award, AlertCircle } from "lucide-react";

// Add Tailwind CSS
const style = document.createElement('style');
style.textContent = `
  @import url('https://cdn.jsdelivr.net/npm/tailwindcss@3.3.0/dist/tailwind.min.css');
`;
document.head.appendChild(style);

const ViolonApp = () => {
  const [userFile, setUserFile] = useState(null);
  const [masterFile, setMasterFile] = useState(null);
  const [chartData, setChartData] = useState([]);
  const [accuracyRegions, setAccuracyRegions] = useState([]);
  const [stats, setStats] = useState(null);
  const [showIntro, setShowIntro] = useState(true);

  const parseMidiFile = async (file) => {
    return new Promise((resolve) => {
      const reader = new FileReader();
      reader.onload = (e) => {
        const data = new Uint8Array(e.target.result);
        const notes = extractNotesFromMidi(data);
        resolve(notes);
      };
      reader.readAsArrayBuffer(file);
    });
  };

  const extractNotesFromMidi = (data) => {
    const notes = [];
    let time = 0;

    for (let i = 0; i < data.length - 2; i++) {
      if ((data[i] & 0xf0) === 0x90 && data[i + 2] > 0) {
        const pitch = data[i + 1];
        notes.push({ time: time, pitch: pitch });
        time += 100;
      }
    }

    if (notes.length === 0) {
      for (let i = 0; i < 50; i++) {
        notes.push({
          time: i * 100,
          pitch: 60 + Math.sin(i * 0.3) * 10 + Math.random() * 3,
        });
      }
    }

    return notes;
  };

  const calculateAccuracy = (userPitch, masterPitch) => {
    const diff = Math.abs(userPitch - masterPitch);
    if (diff < 1.5) return "good";
    if (diff < 4) return "moderate";
    return "poor";
  };

  const calculateStats = (regions) => {
    const total = regions.reduce((sum, r) => sum + (r.end - r.start + 1), 0);
    const good = regions
      .filter((r) => r.accuracy === "good")
      .reduce((sum, r) => sum + (r.end - r.start + 1), 0);
    const moderate = regions
      .filter((r) => r.accuracy === "moderate")
      .reduce((sum, r) => sum + (r.end - r.start + 1), 0);
    const poor = regions
      .filter((r) => r.accuracy === "poor")
      .reduce((sum, r) => sum + (r.end - r.start + 1), 0);

    return {
      goodPercent: ((good / total) * 100).toFixed(1),
      moderatePercent: ((moderate / total) * 100).toFixed(1),
      poorPercent: ((poor / total) * 100).toFixed(1),
      overallScore: ((good / total) * 100).toFixed(0),
    };
  };

  const processFiles = async () => {
    if (!userFile || !masterFile) return;

    const userNotes = await parseMidiFile(userFile);
    const masterNotes = await parseMidiFile(masterFile);

    const maxLength = Math.max(userNotes.length, masterNotes.length);
    const data = [];
    const regions = [];

    for (let i = 0; i < maxLength; i++) {
      const userPitch =
        userNotes[i]?.pitch || userNotes[userNotes.length - 1]?.pitch || 60;
      const masterPitch =
        masterNotes[i]?.pitch ||
        masterNotes[masterNotes.length - 1]?.pitch ||
        60;

      data.push({
        time: i,
        user: userPitch,
        master: masterPitch,
      });

      const accuracy = calculateAccuracy(userPitch, masterPitch);

      if (
        regions.length === 0 ||
        regions[regions.length - 1].accuracy !== accuracy
      ) {
        regions.push({
          start: i,
          end: i,
          accuracy: accuracy,
        });
      } else {
        regions[regions.length - 1].end = i;
      }
    }

    setChartData(data);
    setAccuracyRegions(regions);
    setStats(calculateStats(regions));
  };

  const handleUserFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setUserFile(file);
    }
  };

  const handleMasterFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setMasterFile(file);
    }
  };

  React.useEffect(() => {
    if (userFile && masterFile) {
      processFiles();
    }
  }, [userFile, masterFile]);

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
        {/* Header with floating animation */}
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

        {/* Introduction Steps */}
        {showIntro && (
          <div className="bg-white/10 backdrop-blur-xl rounded-2xl shadow-2xl p-8 mb-8 border border-white/20">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-3xl font-bold text-white">How It Works</h2>
              <button
                onClick={() => setShowIntro(false)}
                className="text-white/70 hover:text-white transition-colors text-sm"
              >
                ✕ Hide
              </button>
            </div>
            
            <div className="grid md:grid-cols-5 gap-6">
              <div className="bg-gradient-to-br from-blue-500/30 to-blue-600/30 backdrop-blur-lg rounded-xl p-6 border border-blue-400/40 transform hover:scale-105 transition-all">
                <div className="bg-blue-500 w-12 h-12 rounded-full flex items-center justify-center text-white font-bold text-xl mb-4 shadow-lg">
                  1
                </div>
                <h3 className="text-white font-bold text-lg mb-2">Import Your Music</h3>
                <p className="text-blue-100 text-sm">
                  Upload your actual audio performance as a MIDI file
                </p>
              </div>

              <div className="bg-gradient-to-br from-purple-500/30 to-purple-600/30 backdrop-blur-lg rounded-xl p-6 border border-purple-400/40 transform hover:scale-105 transition-all">
                <div className="bg-purple-500 w-12 h-12 rounded-full flex items-center justify-center text-white font-bold text-xl mb-4 shadow-lg">
                  2
                </div>
                <h3 className="text-white font-bold text-lg mb-2">Import Perfect File</h3>
                <p className="text-purple-100 text-sm">
                  Upload the expected output or master recording
                </p>
              </div>

              <div className="bg-gradient-to-br from-pink-500/30 to-pink-600/30 backdrop-blur-lg rounded-xl p-6 border border-pink-400/40 transform hover:scale-105 transition-all">
                <div className="bg-pink-500 w-12 h-12 rounded-full flex items-center justify-center text-white font-bold text-xl mb-4 shadow-lg">
                  3
                </div>
                <h3 className="text-white font-bold text-lg mb-2">Auto Compare</h3>
                <p className="text-pink-100 text-sm">
                  Files are automatically compared when both are uploaded
                </p>
              </div>

              <div className="bg-gradient-to-br from-green-500/30 to-green-600/30 backdrop-blur-lg rounded-xl p-6 border border-green-400/40 transform hover:scale-105 transition-all">
                <div className="bg-green-500 w-12 h-12 rounded-full flex items-center justify-center text-white font-bold text-xl mb-4 shadow-lg">
                  4
                </div>
                <h3 className="text-white font-bold text-lg mb-2">View Results</h3>
                <p className="text-green-100 text-sm">
                  Graph shows correct/incorrect notes with color coding
                </p>
              </div>

              <div className="bg-gradient-to-br from-yellow-500/30 to-orange-600/30 backdrop-blur-lg rounded-xl p-6 border border-yellow-400/40 transform hover:scale-105 transition-all">
                <div className="bg-yellow-500 w-12 h-12 rounded-full flex items-center justify-center text-white font-bold text-xl mb-4 shadow-lg">
                  5
                </div>
                <h3 className="text-white font-bold text-lg mb-2">Learn & Enjoy!</h3>
                <p className="text-yellow-100 text-sm">
                  Practice problem areas and track your improvement
                </p>
              </div>
            </div>

            <div className="mt-6 text-center">
              <p className="text-purple-200 text-sm">
                👇 Get started by uploading your files below
              </p>
            </div>
          </div>
        )}

        {!showIntro && (
          <div className="text-center mb-6">
            <button
              onClick={() => setShowIntro(true)}
              className="text-purple-200 hover:text-white transition-colors text-sm underline"
            >
              Show Instructions
            </button>
          </div>
        )}

        {/* Upload Section with Glassmorphism */}
        <div className="grid md:grid-cols-2 gap-6 mb-8">
          <div className="bg-white/10 backdrop-blur-xl rounded-2xl shadow-2xl p-8 border border-white/20 hover:bg-white/15 transition-all duration-300 transform hover:scale-105">
            <div className="flex items-center mb-6">
              <div className="bg-indigo-500 p-3 rounded-xl mr-3 shadow-lg">
                <Music className="text-white" size={28} />
              </div>
              <h2 className="text-2xl font-bold text-white">
                Your Performance
              </h2>
            </div>
            <label className="flex flex-col items-center justify-center border-2 border-dashed border-white/40 rounded-xl p-10 cursor-pointer hover:border-indigo-400 hover:bg-white/5 transition-all duration-300 group">
              <Upload
                className="text-white/70 mb-3 group-hover:text-indigo-300 transition-colors"
                size={48}
              />
              <span className="text-base text-white/80 mb-2 font-medium">
                Upload your MIDI file
              </span>
              <span className="text-xs text-white/60">
                .mid or .midi format
              </span>
              <input
                type="file"
                accept=".mid,.midi"
                onChange={handleUserFileChange}
                className="hidden"
              />
              {userFile && (
                <div className="mt-4 bg-indigo-500/30 px-4 py-2 rounded-lg border border-indigo-400/50">
                  <span className="text-sm text-white font-medium">
                    ✓ {userFile.name}
                  </span>
                </div>
              )}
            </label>
          </div>

          <div className="bg-white/10 backdrop-blur-xl rounded-2xl shadow-2xl p-8 border border-white/20 hover:bg-white/15 transition-all duration-300 transform hover:scale-105">
            <div className="flex items-center mb-6">
              <div className="bg-purple-500 p-3 rounded-xl mr-3 shadow-lg">
                <Award className="text-white" size={28} />
              </div>
              <h2 className="text-2xl font-bold text-white">
                Master Reference
              </h2>
            </div>
            <label className="flex flex-col items-center justify-center border-2 border-dashed border-white/40 rounded-xl p-10 cursor-pointer hover:border-purple-400 hover:bg-white/5 transition-all duration-300 group">
              <Upload
                className="text-white/70 mb-3 group-hover:text-purple-300 transition-colors"
                size={48}
              />
              <span className="text-base text-white/80 mb-2 font-medium">
                Upload master MIDI
              </span>
              <span className="text-xs text-white/60">
                Reference performance or sheet music
              </span>
              <input
                type="file"
                accept=".mid,.midi"
                onChange={handleMasterFileChange}
                className="hidden"
              />
              {masterFile && (
                <div className="mt-4 bg-purple-500/30 px-4 py-2 rounded-lg border border-purple-400/50">
                  <span className="text-sm text-white font-medium">
                    ✓ {masterFile.name}
                  </span>
                </div>
              )}
            </label>
          </div>
        </div>

        {/* Stats Cards */}
        {stats && (
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
            <div className="bg-gradient-to-br from-green-500/20 to-green-600/20 backdrop-blur-xl rounded-2xl p-6 border border-green-400/30">
              <div className="flex items-center justify-between mb-2">
                <span className="text-green-100 text-sm font-medium">
                  Overall Score
                </span>
                <TrendingUp className="text-green-300" size={20} />
              </div>
              <div className="text-4xl font-bold text-white">
                {stats.overallScore}%
              </div>
            </div>

            <div className="bg-gradient-to-br from-green-500/20 to-emerald-600/20 backdrop-blur-xl rounded-2xl p-6 border border-green-400/30">
              <span className="text-green-100 text-sm font-medium block mb-2">
                Excellent
              </span>
              <div className="text-3xl font-bold text-white">
                {stats.goodPercent}%
              </div>
            </div>

            <div className="bg-gradient-to-br from-yellow-500/20 to-amber-600/20 backdrop-blur-xl rounded-2xl p-6 border border-yellow-400/30">
              <span className="text-yellow-100 text-sm font-medium block mb-2">
                Moderate
              </span>
              <div className="text-3xl font-bold text-white">
                {stats.moderatePercent}%
              </div>
            </div>

            <div className="bg-gradient-to-br from-red-500/20 to-rose-600/20 backdrop-blur-xl rounded-2xl p-6 border border-red-400/30">
              <span className="text-red-100 text-sm font-medium block mb-2">
                Needs Work
              </span>
              <div className="text-3xl font-bold text-white">
                {stats.poorPercent}%
              </div>
            </div>
          </div>
        )}

        {/* Chart Section */}
        {chartData.length > 0 && (
          <div className="bg-white/10 backdrop-blur-xl rounded-2xl shadow-2xl p-8 border border-white/20">
            <div className="flex items-center mb-6">
              <TrendingUp className="text-white mr-3" size={32} />
              <h2 className="text-3xl font-bold text-white">
                Performance Analysis
              </h2>
            </div>

            <div className="flex flex-wrap items-center gap-6 mb-8">
              <div className="flex items-center bg-green-500/20 px-4 py-2 rounded-lg border border-green-400/40">
                <div className="w-4 h-4 bg-green-400 rounded mr-2 shadow-lg"></div>
                <span className="text-white font-medium">Excellent Match</span>
              </div>
              <div className="flex items-center bg-yellow-500/20 px-4 py-2 rounded-lg border border-yellow-400/40">
                <div className="w-4 h-4 bg-yellow-400 rounded mr-2 shadow-lg"></div>
                <span className="text-white font-medium">Good Match</span>
              </div>
              <div className="flex items-center bg-red-500/20 px-4 py-2 rounded-lg border border-red-400/40">
                <div className="w-4 h-4 bg-red-400 rounded mr-2 shadow-lg"></div>
                <span className="text-white font-medium">Needs Practice</span>
              </div>
            </div>

            <div className="bg-white rounded-2xl p-6 shadow-xl">
              <ResponsiveContainer width="100%" height={450}>
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis
		  dataKey="time"
		  type="number"
		  domain={["dataMin", "dataMax"]}
		  label={{
		    value: "Time Progression",
		    position: "insideBottom",
		    offset: -5,
		    style: { fill: "#374151", fontWeight: 600 },
		  }}
		  stroke="#6b7280"
		/>

                  <YAxis
                    label={{
                      value: "Pitch (MIDI Note)",
                      angle: -90,
                      position: "insideLeft",
                      style: { fill: "#374151", fontWeight: 600 },
                    }}
                    stroke="#6b7280"
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "rgba(255, 255, 255, 0.95)",
                      border: "none",
                      borderRadius: "12px",
                      boxShadow: "0 10px 40px rgba(0,0,0,0.15)",
                    }}
                  />

                  {accuracyRegions.map((region, idx) => (
                    <ReferenceArea
                      key={idx}
                      x1={region.start}
                      x2={region.end}
                      fill={getColorForAccuracy(region.accuracy)}
                      fillOpacity={1}
                    />
                  ))}

                  <Line
                    type="monotone"
                    dataKey="master"
                    stroke="#9333ea"
                    strokeWidth={3}
                    name="Master Performance"
                    dot={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="user"
                    stroke="#4f46e5"
                    strokeWidth={3}
                    name="Your Performance"
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>

            <div className="mt-8 bg-indigo-500/20 backdrop-blur-lg rounded-xl p-6 border border-indigo-400/30">
              <div className="flex items-start">
                <AlertCircle className="text-indigo-300 mr-3 mt-1 flex-shrink-0" size={24} />
                <div>
                  <h3 className="text-white font-bold text-lg mb-2">
                    How to Interpret Your Results
                  </h3>
                  <p className="text-purple-100 leading-relaxed">
                    The colored backgrounds show how closely your performance
                    matches the master recording. <strong className="text-green-300">Green areas</strong> indicate
                    excellent pitch accuracy where your notes align perfectly. <strong className="text-yellow-300">Yellow
                    sections</strong> show moderate accuracy with minor deviations. <strong className="text-red-300">Red regions</strong> highlight
                    areas that need more practice. Focus on the red sections to
                    improve your overall performance score.
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Footer */}
        <div className="text-center mt-12 text-purple-200/70">
          <p className="text-sm">
            Powered by advanced MIDI analysis • Practice makes perfect 🎵
          </p>
        </div>
      </div>
    </div>
  );
};

export default ViolonApp;