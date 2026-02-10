'use client';

import { useState, useCallback } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { FolderOpen, File, X, Plus, Eye } from 'lucide-react';
import type { ContextFilePreview } from '@/types/rag';

interface ContextFilePickerProps {
  /** Currently selected file paths (ephemeral, session-only). */
  selectedPaths: string[];
  /** Callback when the selection changes. */
  onSelectionChange: (paths: string[]) => void;
  /** Backend API base URL for previewing file contents. */
  apiBaseUrl?: string;
}

/**
 * ContextFilePicker allows users to select files or folders as ephemeral
 * context for LLM-based Q&A. Selected files are NOT stored in any database —
 * they are used only for the current session.
 */
export function ContextFilePicker({
  selectedPaths,
  onSelectionChange,
  apiBaseUrl = 'http://localhost:5167',
}: ContextFilePickerProps) {
  const [previews, setPreviews] = useState<ContextFilePreview[]>([]);
  const [isLoadingPreview, setIsLoadingPreview] = useState(false);

  const handleAddFiles = useCallback(async () => {
    try {
      // Use Tauri's native file dialog via invoke
      const selected = await invoke<string[] | null>('select_context_files');
      if (selected && selected.length > 0) {
        const newPaths = [...selectedPaths];
        for (const path of selected) {
          if (!newPaths.includes(path)) {
            newPaths.push(path);
          }
        }
        onSelectionChange(newPaths);
      }
    } catch (error) {
      console.error('Failed to open file picker:', error);
    }
  }, [selectedPaths, onSelectionChange]);

  const handleAddFolder = useCallback(async () => {
    try {
      // Use Tauri's native folder dialog via invoke
      const selected = await invoke<string | null>('select_context_folder');
      if (selected) {
        if (!selectedPaths.includes(selected)) {
          onSelectionChange([...selectedPaths, selected]);
        }
      }
    } catch (error) {
      console.error('Failed to open folder picker:', error);
    }
  }, [selectedPaths, onSelectionChange]);

  const handleRemovePath = useCallback(
    (pathToRemove: string) => {
      onSelectionChange(selectedPaths.filter((p) => p !== pathToRemove));
      setPreviews((prev) => prev.filter((p) => p.file_path !== pathToRemove));
    },
    [selectedPaths, onSelectionChange]
  );

  const handleClearAll = useCallback(() => {
    onSelectionChange([]);
    setPreviews([]);
  }, [onSelectionChange]);

  const handlePreview = useCallback(async () => {
    if (selectedPaths.length === 0) return;
    setIsLoadingPreview(true);
    try {
      const response = await fetch(`${apiBaseUrl}/read-context-files`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: '', file_paths: selectedPaths }),
      });
      if (response.ok) {
        const data: ContextFilePreview[] = await response.json();
        setPreviews(data);
      }
    } catch (error) {
      console.error('Failed to preview files:', error);
    } finally {
      setIsLoadingPreview(false);
    }
  }, [selectedPaths, apiBaseUrl]);

  const getFilename = (path: string) => {
    const parts = path.replace(/\\/g, '/').split('/');
    return parts[parts.length - 1] || path;
  };

  return (
    <div className="space-y-4">
      <div className="bg-white rounded-lg border border-gray-200 p-6 shadow-sm">
        <h3 className="text-lg font-semibold text-gray-900 mb-2">
          Context Files
        </h3>
        <p className="text-sm text-gray-600 mb-4">
          Select files or folders to provide as context for Q&amp;A. Files are
          read on-the-fly and <strong>never stored</strong> in the database —
          context is session-only and cleared when the app closes.
        </p>

        {/* Action buttons */}
        <div className="flex gap-2 mb-4">
          <button
            onClick={handleAddFiles}
            className="flex items-center gap-2 px-3 py-2 text-sm border border-gray-300 rounded-md hover:bg-gray-100 transition-colors"
          >
            <File className="w-4 h-4" />
            Add Files
          </button>
          <button
            onClick={handleAddFolder}
            className="flex items-center gap-2 px-3 py-2 text-sm border border-gray-300 rounded-md hover:bg-gray-100 transition-colors"
          >
            <FolderOpen className="w-4 h-4" />
            Add Folder
          </button>
          {selectedPaths.length > 0 && (
            <>
              <button
                onClick={handlePreview}
                disabled={isLoadingPreview}
                className="flex items-center gap-2 px-3 py-2 text-sm border border-blue-300 text-blue-700 rounded-md hover:bg-blue-50 transition-colors disabled:opacity-50"
              >
                <Eye className="w-4 h-4" />
                {isLoadingPreview ? 'Loading...' : 'Preview'}
              </button>
              <button
                onClick={handleClearAll}
                className="flex items-center gap-2 px-3 py-2 text-sm border border-red-300 text-red-700 rounded-md hover:bg-red-50 transition-colors ml-auto"
              >
                <X className="w-4 h-4" />
                Clear All
              </button>
            </>
          )}
        </div>

        {/* Selected files list */}
        {selectedPaths.length === 0 ? (
          <div className="p-4 border border-dashed border-gray-300 rounded-lg text-center text-sm text-gray-500">
            <Plus className="w-5 h-5 mx-auto mb-2 text-gray-400" />
            No files selected. Add files or a folder to provide context.
          </div>
        ) : (
          <div className="space-y-2">
            {selectedPaths.map((path) => (
              <div
                key={path}
                className="flex items-center justify-between p-3 border rounded-lg bg-gray-50"
              >
                <div className="flex items-center gap-2 min-w-0">
                  <File className="w-4 h-4 text-gray-500 flex-shrink-0" />
                  <div className="min-w-0">
                    <div className="font-medium text-sm truncate">
                      {getFilename(path)}
                    </div>
                    <div className="text-xs text-gray-500 truncate font-mono">
                      {path}
                    </div>
                  </div>
                </div>
                <button
                  onClick={() => handleRemovePath(path)}
                  className="p-1 text-gray-400 hover:text-red-500 transition-colors flex-shrink-0"
                  title="Remove"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            ))}
          </div>
        )}

        {/* Preview section */}
        {previews.length > 0 && (
          <div className="mt-4 space-y-2">
            <h4 className="text-sm font-medium text-gray-700">
              File Previews
            </h4>
            {previews.map((preview) => (
              <div
                key={preview.file_path}
                className="p-3 border rounded-lg bg-blue-50"
              >
                <div className="flex items-center justify-between mb-1">
                  <span className="text-sm font-medium">{preview.filename}</span>
                  <span className="text-xs text-gray-500">
                    {preview.text_length.toLocaleString()} chars
                  </span>
                </div>
                <p className="text-xs text-gray-600 font-mono whitespace-pre-wrap line-clamp-3">
                  {preview.text_preview}
                </p>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Privacy notice */}
      <div className="bg-green-50 border border-green-200 rounded-lg p-4">
        <p className="text-sm text-green-800">
          🔒 <strong>Privacy:</strong> Selected files are read only when you ask
          a question. Their content is sent directly to the LLM and is never
          stored in any database or vector store.
        </p>
      </div>
    </div>
  );
}
