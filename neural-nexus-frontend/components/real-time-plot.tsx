'use client';
import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

interface RealTimePlotProps {
    maxFitnessData?: (number | null)[] | null;
    avgFitnessData?: (number[] | number | null)[] | null; // Supports number or number[]
    diversityData?: (number | null)[] | null;
}

const RealTimePlot: React.FC<RealTimePlotProps> = ({
    maxFitnessData = [],
    avgFitnessData = [],
    diversityData = []
}) => {
    // 1. Safety check: Ensure we are working with arrays to satisfy TypeScript
    const safeMaxFitness = Array.isArray(maxFitnessData) ? maxFitnessData : [];
    const safeAvgFitness = Array.isArray(avgFitnessData) ? avgFitnessData : [];
    const safeDiversity = Array.isArray(diversityData) ? diversityData : [];

    // 2. Identify objective keys (for NSGA-II dashed lines)
    // We look at the first entry of avgFitnessData to see if it's an array of objectives
    const firstAvg = safeAvgFitness.find(item => Array.isArray(item));
    const objectiveKeys = Array.isArray(firstAvg) 
        ? firstAvg.map((_, idx) => `obj_${idx}`) 
        : [];
    
    const objectiveLabels = ["Accuracy", "Efficiency", "Latency"];

    // 3. Construct Chart Data
    const chartData = safeMaxFitness.map((val, i) => {
        const point: any = { 
            generation: i, 
            maxFitness: val 
        };

        const avgVal = safeAvgFitness[i];
        if (Array.isArray(avgVal)) {
            // Multi-objective mapping
            avgVal.forEach((objVal, objIdx) => {
                point[`obj_${objIdx}`] = objVal;
            });
        } else {
            // Standard single objective mapping
            point.avgFitness = avgVal;
        }

        point.diversity = safeDiversity[i];
        return point;
    });

    if (chartData.length === 0) return null;

    return (
        <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" opacity={0.1} />
                <XAxis dataKey="generation" />
                <YAxis yAxisId="fitness" domain={['auto', 'auto']} />
                <YAxis yAxisId="diversity" orientation="right" hide={safeDiversity.length === 0} />
                <Tooltip />
                <Legend />

                {/* Main Best Fitness Line */}
                <Line
                    yAxisId="fitness"
                    type="monotone"
                    dataKey="maxFitness"
                    name="Best Fitness"
                    stroke="#2563eb"
                    strokeWidth={2}
                    dot={false}
                />

                {/* Standard Avg Fitness (if not NSGA-II) */}
                {!firstAvg && (
                   <Line
                       yAxisId="fitness"
                       type="monotone"
                       dataKey="avgFitness"
                       name="Avg Fitness"
                       stroke="#64748b"
                       strokeDasharray="4 4"
                       dot={false}
                   />
                )}

                {/* NSGA-II Multi-Objective Lines */}
                {objectiveKeys.map((key, index) => (
                    <Line
                        key={key}
                        yAxisId="fitness"
                        type="monotone"
                        dataKey={key}
                        name={objectiveLabels[index] || `Obj ${index}`}
                        stroke={['#10b981', '#f59e0b', '#ef4444'][index % 3]}
                        strokeDasharray="5 5"
                        dot={false}
                    />
                ))}
            </LineChart>
        </ResponsiveContainer>
    );
};

export default RealTimePlot;