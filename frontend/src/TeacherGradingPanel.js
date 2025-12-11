// frontend/src/TeacherGradingPanel.js
import React, { useState } from 'react';

const LocalStatusBadge = ({ status }) => {
  const c =
    {
      pending_ml: { bg: '#fff7e6', color: '#fa8c16', l: 'Processing' },
      ml_scored: { bg: '#e6f7ff', color: '#1890ff', l: 'ML Scored' },
      graded: { bg: '#f6ffed', color: '#52c41a', l: 'Graded' },
    }[status] || { bg: '#eee', color: '#000', l: status };
  return (
    <span
      style={{
        padding: '4px 8px',
        borderRadius: '4px',
        background: c.bg,
        color: c.color,
        border: `1px solid ${c.color}`,
      }}
    >
      {c.l}
    </span>
  );
};

const TeacherGradingPanel = ({ pendingScores, questions, onSubmitGrade, loading }) => {
  const [editing, setEditing] = useState({});

  const getQuestion = (qid) => questions.find((q) => q.id === qid);

  const handleChange = (submissionId, field, value) => {
    setEditing((prev) => ({
      ...prev,
      [submissionId]: {
        ...prev[submissionId],
        [field]: value,
      },
    }));
  };

  const handleSave = async (submissionId) => {
    const edit = editing[submissionId] || {};
    const finalScore = edit.final_score;
    const teacherComment = edit.teacher_comment || '';

    if (finalScore === undefined || finalScore === '' || isNaN(finalScore)) {
      alert('Please enter a valid score.');
      return;
    }

    await onSubmitGrade(submissionId, parseFloat(finalScore), teacherComment);
    setEditing((prev) => {
      const cp = { ...prev };
      delete cp[submissionId];
      return cp;
    });
  };

  return (
    <div className="card" style={{ marginTop: '20px', textAlign: 'left' }}>
      <h3>Pending Submissions to Grade</h3>
      {loading ? (
        <p>Loading...</p>
      ) : pendingScores.length === 0 ? (
        <p>No submissions need grading right now.</p>
      ) : (
        <table className="history-table">
          <thead>
            <tr>
              <th>Submission</th>
              <th>Question</th>
              <th>Student</th>
              <th>Student Answer</th>
              <th>Status</th>
              <th>ML Score</th>
              <th>Final Score</th>
              <th>Comment</th>
              <th>Action</th>
            </tr>
          </thead>
          <tbody>
            {pendingScores.map((s) => {
              const edit = editing[s.submission_id] || {};
              const q = getQuestion(s.question_id);

              return (
                <tr key={s.submission_id}>
                  <td>{s.submission_id}</td>

                  <td>
                    <div style={{ fontWeight: 'bold' }}>
                      {q?.title || `Q#${s.question_id}`}
                    </div>
                    {q?.question_text && (
                      <div
                        style={{
                          fontSize: '0.8rem',
                          color: '#666',
                          marginTop: 2,
                          maxWidth: 260,
                          whiteSpace: 'nowrap',
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                        }}
                        title={q.question_text}
                      >
                        {q.question_text}
                      </div>
                    )}
                  </td>

                  <td>{s.student_id}</td>

                  <td>
                    <div
                      style={{
                        maxWidth: 260,
                        maxHeight: 80,
                        overflow: 'auto',
                        fontSize: '0.85rem',
                        whiteSpace: 'pre-wrap',
                        border: '1px dashed #ddd',
                        padding: '4px 6px',
                      }}
                    >
                      {s.answer_text || '(no answer text)'}
                    </div>
                  </td>

                  <td>
                    <LocalStatusBadge status={s.status} />
                  </td>

                  <td>
                    {s.ml_score != null
                      ? `${s.ml_score} ${s.ml_label ? `(${s.ml_label})` : ''}`
                      : '--'}
                  </td>

                  <td>
                    <input
                      type="number"
                      style={{ width: '80px' }}
                      value={
                        edit.final_score !== undefined
                          ? edit.final_score
                          : s.final_score ?? ''
                      }
                      onChange={(e) =>
                        handleChange(
                          s.submission_id,
                          'final_score',
                          e.target.value
                        )
                      }
                    />
                  </td>

                  <td>
                    <input
                      type="text"
                      style={{ width: '160px' }}
                      placeholder="Comment..."
                      value={
                        edit.teacher_comment !== undefined
                          ? edit.teacher_comment
                          : s.teacher_comment ?? ''
                      }
                      onChange={(e) =>
                        handleChange(
                          s.submission_id,
                          'teacher_comment',
                          e.target.value
                        )
                      }
                    />
                  </td>

                  <td>
                    <button
                      className="btn-submit"
                      onClick={() => handleSave(s.submission_id)}
                    >
                      Save
                    </button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
    </div>
  );
};

export default TeacherGradingPanel;
