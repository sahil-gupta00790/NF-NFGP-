// lib/hooks/useTaskPolling.ts
import { useState, useEffect, useRef, useCallback } from 'react';
import { getTaskStatus } from '@/lib/api'; // Assuming this API function exists
import { toast } from "sonner";

// --- Type Definitions ---
// Add 'HALTED' as a possible status
type TaskStatusType = 'PENDING' | 'STARTED' | 'PROGRESS' | 'SUCCESS' | 'FAILURE' | 'REVOKED' | 'HALTED' | string;

// Matches the structure returned in the 'result' or 'meta' field of Celery task state
interface TaskResultData {
    final_model_path?: string;
    best_fitness?: number | null;
    message?: string;
    error?: string;
    avg_objectives?: number[][] | null;
    // Add fields for the new metric histories
    fitness_history?: number[] | null; // Max fitness history
    avg_fitness_history?: number[] | null;
    diversity_history?: number[] | null;
    // Added based on task return values
    best_hyperparameters?: Record<string, any> | null;
    best_fuzzy_parameters?: Record<string, any> | null;
    status?: string; // Could be included in result/meta, e.g., HALTED_BY_USER
}

// Structure expected from the getTaskStatus API endpoint (adjust if your backend sends differently)
interface TaskStatusResponse {
    task_id: string;
    status: TaskStatusType;
    progress?: number | null;
    info?: TaskResultData | any | null; // Meta data during PROGRESS/REVOKED/HALTED state
    result?: TaskResultData | any | null; // Final result data on SUCCESS/FAILURE
    message?: string | null;
}

// State managed by the hook
interface TaskState {
    taskId: string | null;
    status: TaskStatusType | null;
    progress: number | null;
    result: TaskResultData | null;
    fitnessHistory: number[] | null;
    avgFitnessHistory: number[] | null;
    diversityHistory: number[] | null;
    message: string | null;
    error: string | null;
    isActive: boolean;
    avgObjectives: number[][] | null;
}

const initialState: TaskState = {
    taskId: null, status: null, progress: null, result: null,
    fitnessHistory: null, avgFitnessHistory: null, diversityHistory: null,
    message: null, error: null, isActive: false,avgObjectives: null,
};

// Terminal states that should stop polling
const TERMINAL_STATES: TaskStatusType[] = ['SUCCESS', 'FAILURE', 'REVOKED', 'HALTED'];

export function useTaskPolling(endpoint: 'evolver', intervalMs = 3000) {
    const [taskState, setTaskState] = useState<TaskState>(initialState);
    const intervalRef = useRef<NodeJS.Timeout | null>(null);
    const currentTaskIdRef = useRef<string | null>(null);

    // Function to stop polling interval
    const stopPolling = useCallback(() => {
        if (intervalRef.current) { clearInterval(intervalRef.current); intervalRef.current = null; }
        // Ensure isActive reflects the terminal state correctly
        setTaskState(prev => {
            // If previously active AND not already in a known terminal state, set inactive.
            // If already in a terminal state, keep it as inactive.
            if (prev.isActive && !TERMINAL_STATES.includes(prev.status || '')) {
                return { ...prev, isActive: false };
            }
            return prev; // Otherwise, keep state as is
        });
        currentTaskIdRef.current = null; // Clear tracked task ID
    }, []); // No dependencies needed for this version


    // Function to poll status API
    const pollStatus = useCallback(async () => {
    if (!currentTaskIdRef.current) return;

    try {
        const data: TaskStatusResponse = await getTaskStatus(currentTaskIdRef.current);
        const info = data.info || data.result;
        // Update basic status and progress
        setTaskState(prev => {
    
    return {
        ...prev,
        status: data.status,
        progress: data.progress ?? prev.progress,
        message: data.message || prev.message,
        // Only update these if info exists, otherwise keep previous
        fitnessHistory: info?.fitness_history || prev.fitnessHistory,
        avgFitnessHistory: info?.avg_fitness_history || prev.avgFitnessHistory,
        diversityHistory: info?.diversity_history || prev.diversityHistory,
        avgObjectives: info?.avg_objectives || prev.avgObjectives, 
        result: data.status === 'SUCCESS' ? info : prev.result,
        isActive: !TERMINAL_STATES.includes(data.status)
    };
});
        

        // Handle Errors
        if (data.status === 'FAILURE') {
            const errorMsg = info?.error || data.message || 'Task failed';
            setTaskState(prev => ({ ...prev, error: errorMsg, isActive: false }));
            toast.error(`Evolution Failed: ${errorMsg}`);
            stopPolling();
        }

        // Handle Halted state
        if (data.status === 'HALTED') {
            setTaskState(prev => ({ ...prev, isActive: false }));
            toast.info("Evolution halted by user.");
            stopPolling();
        }

        // Stop polling if we reached a terminal state
        if (TERMINAL_STATES.includes(data.status)) {
            setTaskState(prev => ({ ...prev, isActive: false }));
            stopPolling();
        }

    } catch (err: any) {
        console.error("Polling error:", err);
        // Don't stop polling immediately on one network error, but update state
        setTaskState(prev => ({ ...prev, error: "Connection lost. Retrying..." }));
    }
}, [stopPolling]);// Dependencies for pollStatus

    // Function to start a new task and polling
    const startTask = useCallback((taskId: string) => {
        stopPolling(); // Ensure any previous polling is stopped
        currentTaskIdRef.current = taskId;
        setTaskState(initialState); // Reset to initial state FIRST
        setTaskState(prev => ({ // Then set the new task ID and pending status
            ...prev,
            taskId: taskId,
            status: 'PENDING',
            isActive: true,
            message: 'Task submitted, waiting for status...'
        }));

        // Immediate first poll after a short delay
        const firstPollTimeout = setTimeout(() => { if (currentTaskIdRef.current === taskId) { pollStatus(); } }, 1000); // Reduced delay

        // Setup interval
        if (intervalRef.current) clearInterval(intervalRef.current); // Clear just in case
        intervalRef.current = setInterval(pollStatus, intervalMs);

        // Cleanup timeout if component unmounts before it fires
        // Note: This specific cleanup might be redundant due to stopPolling in unmount effect
        // return () => clearTimeout(firstPollTimeout);

    }, [stopPolling, pollStatus, intervalMs]); // Dependencies for startTask

    // Cleanup interval on component unmount
    useEffect(() => {
        // Return the cleanup function
        return () => {
            stopPolling();
        };
    }, [stopPolling]); // Dependency: stopPolling

    // Function to manually reset state (e.g., after halting or error)
    const resetTaskState = useCallback(() => {
        stopPolling();
        setTaskState(initialState);
    }, [stopPolling]);

    // Return state and control functions
    return { taskState, startTask, resetTaskState }; // Removed stopPolling from return unless needed externally
}
