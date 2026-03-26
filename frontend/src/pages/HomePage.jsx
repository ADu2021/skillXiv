import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import SearchBar from '../components/SearchBar'
import SkillCard from '../components/SkillCard'

export default function HomePage() {
  const [skills, setSkills] = useState([])
  const [featured, setFeatured] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetch('./skills-index.json')
      .then(r => r.json())
      .then(data => {
        // Support both old format (array) and new format ({ skills, globalTags })
        const skillList = Array.isArray(data) ? data : (data.skills || [])
        setSkills(skillList)
        // Pick 6 featured skills (diverse sample)
        const picks = []
        const step = Math.floor(skillList.length / 6)
        for (let i = 0; i < 6; i++) {
          picks.push(skillList[i * step + Math.floor(Math.random() * step)])
        }
        setFeatured(picks)
        setLoading(false)
      })
      .catch(() => setLoading(false))
  }, [])

  return (
    <div className="home">
      {/* Hero */}
      <section className="hero">
        <div className="hero-inner">
          <h1 className="hero-title">
            The Open Repository of<br />
            <span className="hero-accent">Agent Skills</span> from Research Papers
          </h1>
          <p className="hero-subtitle">
            SkillXiv is a community-driven repository of agent-ready skills
            extracted from research papers — ready for any AI agent, out of the box.
          </p>
          <SearchBar large />
          <div className="hero-stats">
            <div className="stat">
              <span className="stat-number">{skills.length.toLocaleString()}+</span>
              <span className="stat-label">Skills Extracted</span>
            </div>
            <div className="stat-divider" />
            <div className="stat">
              <span className="stat-number">arXiv</span>
              <span className="stat-label">Paper Source</span>
            </div>
            <div className="stat-divider" />
            <div className="stat">
              <span className="stat-number">Any Agent</span>
              <span className="stat-label">Compatible</span>
            </div>
          </div>
          <a
            href="https://github.com/adu2021/skillxiv"
            target="_blank"
            rel="noopener noreferrer"
            className="github-repo-btn"
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
              <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23A11.509 11.509 0 0 1 12 5.803c1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576C20.566 21.797 24 17.3 24 12c0-6.627-5.373-12-12-12z"/>
            </svg>
            GitHub Repo
          </a>
        </div>
      </section>

      {/* How it works */}
      <section className="how-it-works">
        <div className="section-inner">
          <h2 className="section-title">How It Works</h2>
          <div className="steps">
            <div className="step">
              <div className="step-number">1</div>
              <h3>Paper Ingestion</h3>
              <p>We continuously index papers from arXiv across ML, AI, robotics, and more.</p>
            </div>
            <div className="step-arrow">→</div>
            <div className="step">
              <div className="step-number">2</div>
              <h3>Skill Extraction</h3>
              <p>Our paper2skill engines distill each paper into a structured, actionable SKILL.md file.</p>
            </div>
            <div className="step-arrow">→</div>
            <div className="step">
              <div className="step-number">3</div>
              <h3>Agent-Ready</h3>
              <p>Skills are immediately usable by any AI agent — just plug in and go.</p>
            </div>
          </div>
        </div>
      </section>

      {/* Featured skills */}
      <section className="featured">
        <div className="section-inner">
          <div className="section-header">
            <h2 className="section-title">Featured Skills</h2>
            <Link to="/browse" className="section-link">Browse all →</Link>
          </div>
          {loading ? (
            <div className="loading">Loading skills...</div>
          ) : (
            <div className="skills-grid">
              {featured.map(skill => (
                <SkillCard key={skill.id} skill={skill} />
              ))}
            </div>
          )}
        </div>
      </section>
    </div>
  )
}
