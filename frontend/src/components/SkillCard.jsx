import { Link } from 'react-router-dom'

export default function SkillCard({ skill }) {
  // Truncate description to ~120 chars
  const shortDesc = skill.description && skill.description.length > 140
    ? skill.description.slice(0, 140).replace(/\s+\S*$/, '') + '...'
    : skill.description

  return (
    <Link to={`/skill/${skill.id}`} className="skill-card">
      <div className="skill-card-header">
        <span className="skill-card-engine">{skill.engine || 'v0.1'}</span>
      </div>
      <h3 className="skill-card-name">{skill.name}</h3>
      <p className="skill-card-desc">{shortDesc}</p>
      {skill.paperTitle && (
        <div className="skill-card-paper">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
            <polyline points="14 2 14 8 20 8"/>
            <line x1="16" y1="13" x2="8" y2="13"/>
            <line x1="16" y1="17" x2="8" y2="17"/>
          </svg>
          <span>{skill.paperTitle}</span>
        </div>
      )}
    </Link>
  )
}
