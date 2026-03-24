import { Routes, Route } from 'react-router-dom'
import Header from './components/Header'
import Footer from './components/Footer'
import HomePage from './pages/HomePage'
import SkillPage from './pages/SkillPage'
import BrowsePage from './pages/BrowsePage'

export default function App() {
  return (
    <div className="app">
      <Header />
      <main className="main-content">
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/browse" element={<BrowsePage />} />
          <Route path="/skill/:skillId" element={<SkillPage />} />
        </Routes>
      </main>
      <Footer />
    </div>
  )
}
