import { useState, useEffect } from 'react'
import { useParams, Link } from 'react-router-dom'
import { marked } from 'marked'
import hljs from 'highlight.js'

// Configure marked with highlight.js
marked.setOptions({
  highlight: function(code, lang) {
    if (lang && hljs.getLanguage(lang)) {
      try {
        return hljs.highlight(code, { language: lang }).value
      } catch {}
    }
    return hljs.highlightAuto(code).value
  },
  breaks: false,
  gfm: true
})

export default function SkillPage() {
  const { skillId } = useParams()
  const [skill, setSkill] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    setLoading(true)
    fetch(`./skills-data/${skillId}.json`)
      .then(r => {
        if (!r.ok) throw new Error('Skill not found')
        return r.json()
      })
      .then(data => {
        setSkill(data)
        setLoading(false)
      })
      .catch(err => {
        setError(err.message)
        setLoading(false)
      })
  }, [skillId])

  useEffect(() => {
    window.scrollTo(0, 0)
  }, [skillId])

  if (loading) return <div className="skill-page"><div className="section-inner"><div className="loading">Loading skill...</div></div></div>
  if (error) return (
    <div className="skill-page">
      <div className="section-inner">
        <div className="no-results">
          <h3>Skill not found</h3>
          <p>{error}</p>
          <Link to="/browse" className="back-link">← Back to Browse</Link>
        </div>
      </div>
    </div>
  )

  const html = marked.parse(skill.content)

  return (
    <div className="skill-page">
      <div className="section-inner">
        <Link to="/browse" className="back-link">← Back to Browse</Link>
        <div className="skill-detail">
          <aside className="skill-meta">
            <div className="meta-card">
              <h3>Skill Info</h3>
              {skill.name && (
                <div className="meta-row">
                  <span className="meta-label">Name</span>
                  <span className="meta-value"><code>{skill.name}</code></span>
                </div>
              )}
              {skill.engine && (
                <div className="meta-row">
                  <span className="meta-label">Engine</span>
                  <span className="meta-value">{skill.engine}</span>
                </div>
              )}
              {skill.paperTitle && (
                <div className="meta-row">
                  <span className="meta-label">Paper Title</span>
                  <span className="meta-value">{skill.paperTitle}</span>
                </div>
              )}
              {skill.url && (
                <div className="meta-row">
                  <span className="meta-label">Link</span>
                  <span className="meta-value"><a href={skill.url} target="_blank" rel="noopener noreferrer">{skill.url}</a></span>
                </div>
              )}
              {skill.keywords && (
                <div className="meta-row">
                  <span className="meta-label">Keywords</span>
                  <span className="meta-value meta-keywords">{skill.keywords}</span>
                </div>
              )}
              {skill.description && (
                <div className="meta-row">
                  <span className="meta-label">Description</span>
                  <span className="meta-value meta-desc">{skill.description}</span>
                </div>
              )}
            </div>
          </aside>
          <article className="skill-content" dangerouslySetInnerHTML={{ __html: html }} />
        </div>
      </div>
    </div>
  )
}
