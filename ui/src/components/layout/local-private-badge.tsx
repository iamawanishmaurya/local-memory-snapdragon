import { useEffect, useState } from 'react'
import { api } from '@/lib/api'
import { Badge } from '@/components/ui/badge'

/**
 * "Local · Private" badge (D-06) — visible proof the local auth token works.
 * Renders only after the first SUCCESSFUL authenticated /api/status call; if
 * the server is down or unauthenticated, status fails and the badge never
 * shows.
 */
export function LocalPrivateBadge({ className }: { className?: string }) {
  const [authenticated, setAuthenticated] = useState(false)

  useEffect(() => {
    let cancelled = false
    api
      .status()
      .then(() => {
        if (!cancelled) setAuthenticated(true)
      })
      .catch(() => {
        /* status failed (server down / unauthenticated) — badge stays hidden */
      })
    return () => {
      cancelled = true
    }
  }, [])

  if (!authenticated) return null
  return (
    <Badge
      variant='outline'
      className={`gap-1.5 border-emerald-500/40 bg-emerald-500/10 text-emerald-700 dark:text-emerald-400 ${className ?? ''}`}
      aria-label='All data stays local and private'
    >
      <span className='size-2 rounded-full bg-emerald-500' aria-hidden />
      Local · Private
    </Badge>
  )
}
