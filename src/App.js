import React, { useState } from "react";
import { analyzePerformanceWithLLM } from "./ai";
import { compareMidiFiles } from "./backend";
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

const NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
const MAX_DIFF_SAMPLES = 250;
const MAX_REGION_SUMMARY = 12;

const midiToNote = (pitch = 60) => {
  if (typeof pitch !== "number" || Number.isNaN(pitch)) {
    return "N/A";
  }
  const name = NOTE_NAMES[pitch % 12] || "C";
  const octave = Math.floor(pitch / 12) - 1;
  return `${name}${octave}`;
};

const buildDiffPayload = (stats, regions, diffRows) => {
  if (!stats || !regions?.length || !diffRows?.length) {
    return "";
  }

  const normalizedRegions = regions.slice(0, MAX_REGION_SUMMARY).map((region) => ({
    ...region,
    length: region.end - region.start + 1,
  }));

  const normalizedDiffs = diffRows.slice(0, MAX_DIFF_SAMPLES);

  return JSON.stringify(
    {
      stats,
      regions: normalizedRegions,
      diffSamples: normalizedDiffs,
    },
    null,
    2
  );
};

// Add Tailwind CSS
const style = document.createElement('style');
style.textContent = `
  @import url('https://cdn.jsdelivr.net/npm/tailwindcss@3.3.0/dist/tailwind.min.css');
`;
document.head.appendChild(style);

const accuracyBucket = (label) => {
  if (label === "good") return "good";
  if (label === "moderate") return "moderate";
  return "poor";
};

const accuracyFromEvaluation = (evaluation) => {
  if (!evaluation) return "missing";
  if (evaluation.student_error_type === "ok") return "good";
  if (evaluation.student_error_type === "wrong_duration") return "moderate";
  return "poor";
};

const buildRegionsFromChart = (chartEntries) => {
  if (!chartEntries?.length) return [];
  const regions = [];
  chartEntries.forEach((entry, index) => {
    const bucket = accuracyBucket(entry.accuracy);
    if (!regions.length || regions[regions.length - 1].accuracy !== bucket) {
      regions.push({ start: index, end: index, accuracy: bucket });
    } else {
      regions[regions.length - 1].end = index;
    }
  });
  return regions;
};

const calculateStatsFromRegions = (regions) => {
  if (!regions.length) return null;
  const total = regions.reduce((sum, r) => sum + (r.end - r.start + 1), 0);
  if (!total) return null;
  const countByAccuracy = regions.reduce(
    (acc, region) => {
      acc[region.accuracy] = (acc[region.accuracy] || 0) + (region.end - region.start + 1);
      return acc;
    },
    { good: 0, moderate: 0, poor: 0 }
  );

  const pct = (count) => ((count / total) * 100).toFixed(1);
  return {
    goodPercent: pct(countByAccuracy.good || 0),
    moderatePercent: pct(countByAccuracy.moderate || 0),
    poorPercent: pct(countByAccuracy.poor || 0),
    overallScore: ((countByAccuracy.good || 0) / total * 100).toFixed(0),
  };
};

const buildDiffRows = (result) => {
  const rows = (result?.error_report || []).map((entry, index) => {
    const masterPitch = entry.gt_pitch;
    const studentPitch = entry.student_pitch;
    const pitchDelta =
      typeof studentPitch === "number" && typeof masterPitch === "number"
        ? Number((studentPitch - masterPitch).toFixed(2))
        : null;

    return {
      index,
      approxTimeMs: Math.round((entry.time_sec || 0) * 1000),
      masterPitch,
      masterNote: typeof masterPitch === "number" ? midiToNote(masterPitch) : null,
      userPitch: studentPitch,
      userNote: typeof studentPitch === "number" ? midiToNote(studentPitch) : null,
      pitchDelta,
      accuracy: entry.student_error_type,
      onsetDiffSec: entry.onset_diff_sec ?? null,
      offsetDiffSec: entry.offset_diff_sec ?? null,
    };
  });
  return rows;
};

const buildChartDataFromResult = (result) => {
  const refNotes = result?.reference_notes || [];
  const studentNotes = result?.student_notes_warped || [];
  const evalByRef = new Map();
  (result?.match_evaluations || []).forEach((evaluation) => {
    evalByRef.set(evaluation.reference_index, evaluation);
  });

  const chart = refNotes.map((note, index) => {
    const evaluation = evalByRef.get(index);
    const studentNote = evaluation ? studentNotes[evaluation.student_index] : null;

    return {
      time: note.start,
      master: note.pitch,
      user: studentNote?.pitch ?? null,
      accuracy: evaluation ? accuracyFromEvaluation(evaluation) : "poor",
    };
  });

  (result?.extra_student_indices || []).forEach((studentIndex) => {
    const note = studentNotes[studentIndex];
    if (!note) return;
    chart.push({
      time: note.start,
      master: null,
      user: note.pitch,
      accuracy: "poor",
    });
  });

  chart.sort((a, b) => a.time - b.time);
  return chart;
};

const deriveVisualizationState = (result) => {
  if (!result) {
    return { chartData: [], accuracyRegions: [], stats: null, diffRows: [] };
  }

  const chartData = buildChartDataFromResult(result);
  const accuracyRegions = buildRegionsFromChart(chartData);
  const stats = calculateStatsFromRegions(accuracyRegions);
  const diffRows = buildDiffRows(result);
  return { chartData, accuracyRegions, stats, diffRows };
};

const ViolonApp = () => {
  const [userFile, setUserFile] = useState(null);
  const [masterFile, setMasterFile] = useState(null);
  const [chartData, setChartData] = useState([]);
  const [accuracyRegions, setAccuracyRegions] = useState([]);
  const [stats, setStats] = useState(null);
  const [showIntro, setShowIntro] = useState(true);
  const [aiFeedback, setAiFeedback] = useState("");
  const [aiError, setAiError] = useState(null);
  const [aiLoading, setAiLoading] = useState(false);
  const [analysisLoading, setAnalysisLoading] = useState(false);
  const [analysisError, setAnalysisError] = useState(null);


  const generateAiFeedback = React.useCallback(async (diffJson) => {
    if (!diffJson) {
        setAiFeedback("");
        return;
      }

      setAiLoading(true);
      setAiError(null);

      try {
      const response = await analyzePerformanceWithLLM({
        diffJson,
      });
        setAiFeedback(response);
      } catch (error) {
        console.error("AI feedback error:", error);
        setAiError(
          error?.message ||
            "We couldn't fetch coaching feedback right now. Try again later."
        );
      } finally {
        setAiLoading(false);
      }
  }, []);

  const processFiles = React.useCallback(async () => {
    if (!userFile || !masterFile) return;

    setAnalysisLoading(true);
    setAnalysisError(null);

    try {
      const apiResult = await compareMidiFiles({
        referenceFile: masterFile,
        studentFile: userFile,
      });

      const { chartData: chart, accuracyRegions: regions, stats: derivedStats, diffRows } =
        deriveVisualizationState(apiResult);

      setChartData(chart);
      setAccuracyRegions(regions);
      setStats(derivedStats);

      const diffJsonPayload = buildDiffPayload(derivedStats, regions, diffRows);
      generateAiFeedback(diffJsonPayload);
    } catch (error) {
      console.error("Comparison error:", error);
      setAnalysisError(
        error?.message || "Unable to analyze the uploaded MIDI files right now."
      );
      setChartData([]);
      setAccuracyRegions([]);
      setStats(null);
      setAiFeedback("");
    } finally {
      setAnalysisLoading(false);
    }
  }, [userFile, masterFile, generateAiFeedback]);

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
    processFiles();
  }, [processFiles]);

  React.useEffect(() => {
    if (!userFile || !masterFile) {
      setAiFeedback("");
      setAiError(null);
      setAiLoading(false);
      setAnalysisError(null);
      setChartData([]);
      setAccuracyRegions([]);
      setStats(null);
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
        {analysisLoading && (
          <div className="bg-white/10 backdrop-blur-xl rounded-2xl shadow-2xl p-6 border border-white/30 text-center text-white mb-6">
            <p className="text-lg font-semibold">Analyzing your performance...</p>
            <p className="text-sm text-purple-100 mt-2">This may take a few seconds depending on file size.</p>
          </div>
        )}

        {analysisError && (
          <div className="bg-red-500/20 backdrop-blur-xl rounded-2xl shadow-2xl p-4 border border-red-400/40 text-white mb-6">
            <p className="font-semibold">{analysisError}</p>
          </div>
        )}

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

        {(aiLoading || aiFeedback || aiError) && (
          <div className="bg-white/10 backdrop-blur-xl rounded-2xl shadow-2xl p-8 border border-white/20 mb-8">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h2 className="text-3xl font-bold text-white">Violon AI Coach</h2>
                <p className="text-purple-100 text-sm">
                  Personalized practice guidance
                </p>
              </div>
              {aiLoading && (
                <span className="text-white/70 text-sm animate-pulse">
                  Analyzing...
                </span>
              )}
            </div>

            {aiError ? (
              <div className="bg-red-500/20 border border-red-400/40 text-red-100 text-sm p-4 rounded-xl">
                {aiError}
              </div>
            ) : (
              <div className="bg-indigo-900/30 border border-indigo-400/30 rounded-xl p-5">
                {aiLoading ? (
                  <p className="text-white/80">
                    Comparing your performance with the master reference...
                  </p>
                ) : (
                  <p className="text-white/90 whitespace-pre-wrap font-mono text-sm">
                    {aiFeedback ||
                      "Upload both MIDI files to receive personalized coaching feedback."}
                  </p>
                )}
              </div>
            )}

            <p className="text-white/50 text-xs mt-3">
              Powered by Gemini
            </p>
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

                  {accuracyRegions.map((region, idx) => {
                    const startTime = chartData[region.start]?.time ?? 0;
                    const endTime = chartData[region.end]?.time ?? startTime;
                    return (
                      <ReferenceArea
                        key={idx}
                        x1={startTime}
                        x2={endTime}
                        fill={getColorForAccuracy(region.accuracy)}
                        fillOpacity={1}
                      />
                    );
                  })}

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
