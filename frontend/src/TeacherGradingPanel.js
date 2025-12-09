// frontend/src/TeacherGradingPanel.js
import React, { useState } from 'react';

// 这里单独搞一个本地的 StatusBadge，不跟学生那边的混用也没问题
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

/**
 * 老师改分面板：
 * - pendingScores: 后端 /scores/pending 返回的数组
 * - onSubmitGrade: (submissionId, finalScore, comment) => Promise
 * - loading: 是否在加载中（可以绑定到按钮 / 文本）
 */
const TeacherGradingPanel = ({ pendingScores, onSubmitGrade, loading }) => {
  const [editing, setEditing] = useState({}); // { [submission_id]: { final_score, teacher_comment } }

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

    // 保存成功后，清掉本行的临时编辑状态
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
              <th>Submission ID</th>
              <th>Question ID</th>
              <th>Student ID</th>
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
              return (
                <tr key={s.submission_id}>
                  <td>{s.submission_id}</td>
                  <td>{s.question_id}</td>
                  <td>{s.student_id}</td>
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
