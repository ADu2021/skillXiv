/**
 * Horizontal scrollable tag filter bar for the Browse page.
 * Shows tags sorted by count, highlights the active tag.
 */
export default function TagBar({ tags, activeTag, onTagClick }) {
  if (!tags || tags.length === 0) return null

  return (
    <div className="tag-bar">
      <div className="tag-bar-scroll">
        {tags.map(tag => (
          <button
            key={tag.slug}
            className={`tag-pill${activeTag === tag.slug ? ' tag-pill-active' : ''}`}
            onClick={() => onTagClick(tag.slug)}
            title={`${tag.count} skill${tag.count !== 1 ? 's' : ''}`}
          >
            {tag.name}
            <span className="tag-pill-count">{tag.count}</span>
          </button>
        ))}
      </div>
    </div>
  )
}
