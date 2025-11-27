import React, { useState } from 'react';
import './App.css'; // 引入上面的样式

// 模拟初始数据
const INITIAL_DATA = [
  { id: 1, subject: 'ee547', name: 'hw1', date: '2023-10-01', status: 'graded', score: 95 },
  { id: 2, subject: 'ee641', name: 'hw2', date: '2023-10-05', status: 'graded', score: 88 },
  { id: 3, subject: 'ee541', name: 'hw3', date: '2023-10-10', status: 'pending', score: null }
];

function App() {
  // 状态管理
  const [homeworkList, setHomeworkList] = useState(INITIAL_DATA);

  // 添加作业的处理函数
  const handleAddHomework = (newHomework) => {
    // 将新作业添加到列表头部
    setHomeworkList([newHomework, ...homeworkList]);
  };

  return (
    <div className="container">
      <Header />

      <div className="main-grid">
        {/**/}
        <UploadCard onUpload={handleAddHomework} />

        {/**/}
        <HistoryList homeworks={homeworkList} />
      </div>
    </div>
  );
}

// --- 子组件：顶部导航 ---
const Header = () => (
  <header>
    <h1>Student Homework</h1>
    <div className="user-info">
      <span>Welcome</span>
      <div className="avatar">李</div>
    </div>
  </header>
);

// --- 子组件：上传卡片 ---
const UploadCard = ({ onUpload }) => {
  const [subject, setSubject] = useState('');
  const [name, setName] = useState('');
  const [file, setFile] = useState(null);
  const [isUploading, setIsUploading] = useState(false);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!file || !subject || !name) return alert("please fill correct info");

    setIsUploading(true);

    setTimeout(() => {
      const newEntry = {
        id: Date.now(),
        subject,
        name,
        date: new Date().toISOString().split('T')[0],
        status: 'pending',
        score: null
      };

      onUpload(newEntry);

      // 重置表单
      setSubject('');
      setName('');
      setFile(null);
      setIsUploading(false);
      alert("success");
    }, 1000);
  };

  return (
    <section className="card">
      <h2>upload new homework</h2>
      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label>course</label>
          <select
            value={subject}
            onChange={(e) => setSubject(e.target.value)}
            required
          >
            <option value="" disabled>select...</option>
            <option value="ee541">ee541</option>
            <option value="ee503">ee503</option>
            <option value="ee641">ee641</option>
            <option value="ee547">ee547</option>
          </select>
        </div>

        <div className="form-group">
          <label>homework name</label>
          <input
            type="text"
            placeholder=""
            value={name}
            onChange={(e) => setName(e.target.value)}
            required
          />
        </div>

        <div className="form-group">
          <label>file</label>
          <div className="file-drop-area">
            <span style={{ color: file ? 'green' : '#888' }}>
              {file ? `selected: ${file.name}` : 'upload file'}
            </span>
            <input
              type="file"
              className="file-input-hidden"
              onChange={handleFileChange}
              required
            />
          </div>
        </div>

        <button type="submit" className="btn-submit" disabled={isUploading}>
          {isUploading ? 'uploading' : 'submit'}
        </button>
      </form>
    </section>
  );
};

// --- 子组件：历史记录列表 ---
const HistoryList = ({ homeworks }) => {
  return (
    <section className="card">
      <h2>history</h2>
      {homeworks.length === 0 ? (
        <p style={{textAlign:'center', color:'#999'}}>empty</p>
      ) : (
        <table>
          <thead>
            <tr>
              <th>Infomation</th>
              <th>submit time</th>
              <th>status</th>
              <th>score</th>
            </tr>
          </thead>
          <tbody>
            {homeworks.map((item) => (
              <tr key={item.id}>
                <td>
                  <div style={{ fontWeight: 'bold' }}>{item.subject}</div>
                  <div style={{ fontSize: '0.85rem', color: '#666' }}>{item.name}</div>
                </td>
                <td>{item.date}</td>
                <td>
                  <StatusBadge status={item.status} />
                </td>
                <td>
                  {item.status === 'graded' ? (
                    <span style={{ fontWeight: 'bold' }}>{item.score} 分</span>
                  ) : (
                    <span style={{ color: '#ccc' }}>--</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  );
};

const StatusBadge = ({ status }) => {
  if (status === 'graded') {
    return <span className="badge badge-graded">Corrected</span>;
  }
  return <span className="badge badge-pending">Pending</span>;
};

export default App;