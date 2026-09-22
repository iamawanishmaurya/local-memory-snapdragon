import { useState } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import {
  FileText,
  Image as ImageIcon,
  File,
  BookOpen,
  Loader2,
  SearchIcon,
  Zap,
  FolderOpen,
  Pause,
  Play,
  ExternalLink,
} from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Skeleton } from '@/components/ui/skeleton'
import { Header } from '@/components/layout/header'
import { Main } from '@/components/layout/main'
import { ProfileDropdown } from '@/components/profile-dropdown'
import { ThemeSwitch } from '@/components/theme-switch'
import { Search as GlobalSearch } from '@/components/search'
import { ConfigDrawer } from '@/components/config-drawer'
import { api, fmtBytes, fmtDate, type SearchResult, type Status } from '@/lib/api'

const EXAMPLES = ['pen with blue book', 'invoice from last month', 'error screenshots']

const KIND_ICONS: Record<string, React.ReactNode> = {
  text: <FileText className='size-5' />,
  pdf: <BookOpen className='size-5' />,
  docx: <FileText className='size-5' />,
  image: <ImageIcon className='size-5' />,
  other: <File className='size-5' />,
}

export function SearchPage() {
  const [query, setQuery] = useState('')
  const [submitted, setSubmitted] = useState('')

  const status = useQuery({
    queryKey: ['status'],
    queryFn: api.status,
    refetchInterval: 3000,
  })
  const search = useMutation({
    mutationFn: (q: string) => api.search(q),
  })

  const doSearch = (q: string) => {
    const trimmed = q.trim()
    if (!trimmed) return
    setQuery(trimmed)
    setSubmitted(trimmed)
    search.mutate(trimmed)
  }

  const s: Status | undefined = status.data
  const indexing = s?.pipeline.status === 'indexing'
  const results: SearchResult[] = search.data?.results ?? []

  return (
    <>
      <Header>
        <GlobalSearch className='me-auto' />
        {s?.paused ? (
          <Button size='sm' variant='outline' onClick={() => api.resume().then(() => status.refetch())}>
            <Play /> Resume indexing
          </Button>
        ) : (
          <Button size='sm' variant='outline' onClick={() => api.pause().then(() => status.refetch())}>
            <Pause /> Pause indexing
          </Button>
        )}
        <ThemeSwitch />
        <ConfigDrawer />
        <ProfileDropdown />
      </Header>

      <Main fixed>
        <div className='mb-4 flex flex-wrap items-center gap-2'>
          <h1 className='text-2xl font-bold tracking-tight md:text-3xl'>Search</h1>
          <Badge
            variant={s?.npu.qnn_available ? 'default' : 'secondary'}
            className='gap-1'
          >
            <Zap className='size-3' />
            {s?.npu.qnn_available ? 'NPU: QNN active' : 'CPU fallback'}
          </Badge>
          {indexing && (
            <Badge variant='secondary' className='gap-1'>
              <Loader2 className='size-3 animate-spin' />
              Indexing {s!.pipeline.done}/{s!.pipeline.queued}
            </Badge>
          )}
          {s?.paused && <Badge variant='destructive'>Paused</Badge>}
        </div>

        {/* Search bar */}
        <form
          className='flex gap-2'
          onSubmit={(e) => {
            e.preventDefault()
            doSearch(query)
          }}
        >
          <Input
            autoFocus
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search everything… e.g. 'invoice from last month'"
            className='h-11 text-base'
          />
          <Button type='submit' className='h-11' disabled={search.isPending}>
            {search.isPending ? <Loader2 className='animate-spin' /> : <SearchIcon />}
            Search
          </Button>
        </form>
        <div className='mt-2 flex flex-wrap items-center gap-2 text-sm text-muted-foreground'>
          Try:
          {EXAMPLES.map((ex) => (
            <button
              key={ex}
              onClick={() => doSearch(ex)}
              className='underline decoration-dotted underline-offset-4 hover:text-foreground'
            >
              {ex}
            </button>
          ))}
        </div>

        {/* Empty / no-folder states */}
        {s && s.watched_folders.length === 0 && (
          <Card className='mt-6 border-dashed'>
            <CardContent className='flex items-center gap-3 pt-6'>
              <FolderOpen className='size-6 text-muted-foreground' />
              <div>
                <p className='font-medium'>No folders indexed yet</p>
                <p className='text-sm text-muted-foreground'>
                  Add Downloads, Documents or Desktop on the Folders page to start building your memory.
                </p>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Stats strip */}
        {s && s.watched_folders.length > 0 && (
          <div className='mt-6 grid gap-4 sm:grid-cols-3'>
            <Card>
              <CardHeader className='pb-2'>
                <CardTitle className='text-sm font-medium text-muted-foreground'>Files indexed</CardTitle>
              </CardHeader>
              <CardContent>
                <div className='text-2xl font-bold'>{s.stats.files.toLocaleString()}</div>
                <p className='text-xs text-muted-foreground'>{fmtBytes(s.stats.total_bytes)} across {s.watched_folders.length} folder(s)</p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className='pb-2'>
                <CardTitle className='text-sm font-medium text-muted-foreground'>Text chunks</CardTitle>
              </CardHeader>
              <CardContent>
                <div className='text-2xl font-bold'>{s.stats.chunks.toLocaleString()}</div>
                <p className='text-xs text-muted-foreground'>embedded with Nomic-Embed-Text</p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className='pb-2'>
                <CardTitle className='text-sm font-medium text-muted-foreground'>Images understood</CardTitle>
              </CardHeader>
              <CardContent>
                <div className='text-2xl font-bold'>{s.stats.images.toLocaleString()}</div>
                <p className='text-xs text-muted-foreground'>OCR + CLIP vision embeddings</p>
              </CardContent>
            </Card>
          </div>
        )}

        {/* Results */}
        {submitted && (
          <div className='mt-6 space-y-3'>
            <h2 className='text-sm font-medium text-muted-foreground'>
              {search.isPending
                ? 'Searching…'
                : `${results.length} result${results.length === 1 ? '' : 's'} for “${submitted}”`}
            </h2>
            {search.isPending &&
              Array.from({ length: 3 }).map((_, i) => <Skeleton key={i} className='h-24 w-full' />)}
            {!search.isPending && results.length === 0 && (
              <p className='py-8 text-center text-muted-foreground'>
                No results. Try different words, or if indexing just started, give it a moment and
                try again.
              </p>
            )}
            {results.map((r) => (
              <ResultRow key={r.file_id} r={r} />
            ))}
          </div>
        )}
      </Main>
    </>
  )
}

const VIA_LABEL: Record<NonNullable<SearchResult['matched_via']>, string> = {
  semantic: 'via semantic',
  keyword: 'via keyword',
  both: 'via both',
}

const VIA_VARIANT: Record<
  NonNullable<SearchResult['matched_via']>,
  'default' | 'secondary' | 'outline'
> = {
  semantic: 'default',
  keyword: 'outline',
  both: 'default',
}

/** Split a snippet containing FTS5 <mark>…</mark> into text/highlight spans.
 * Plain string splitting — no dangerouslySetInnerHTML (Phase 2 security posture). */
function MarkedSnippet({ text }: { text: string }) {
  const parts = text.split(/(<mark>|<\/mark>)/)
  let marked = false
  return (
    <>
      {parts.map((p, i) => {
        if (p === '<mark>') {
          marked = true
          return null
        }
        if (p === '</mark>') {
          marked = false
          return null
        }
        if (!p) return null
        return marked ? (
          <mark key={i} className='rounded bg-yellow-200/60 px-0.5 text-foreground dark:bg-yellow-500/30'>
            {p}
          </mark>
        ) : (
          <span key={i}>{p}</span>
        )
      })}
    </>
  )
}

/** D-06 "why" line prefix: images visibly quote their OCR text on stage. */
function snippetPrefix(source: SearchResult['snippet_source']): string {
  if (source === 'ocr') return 'OCR text: '
  if (source === 'name') return 'matched on filename: '
  return ''
}

function ResultRow({ r }: { r: SearchResult }) {
  const via = r.matched_via
  const snippetBody = r.highlight ?? r.snippet
  const [openState, setOpenState] = useState<'idle' | 'pending' | 'error'>('idle')

  const openFile = async () => {
    setOpenState('pending')
    try {
      await api.openFile(r.file_id)
      setOpenState('idle')
    } catch {
      // 404 / 410 / network error: brief inline failure state, no toast spam.
      setOpenState('error')
    }
  }

  return (
    <Card className='flex gap-4 p-4'>
      {r.kind === 'image' ? (
        <img
          src={`/api/thumbnail?file_id=${r.file_id}`}
          onError={(e) => {
            ;(e.target as HTMLImageElement).replaceWith(kindFallback(r.kind))
          }}
          alt={r.name}
          className='size-20 flex-none rounded-lg object-cover'
        />
      ) : (
        <div className='flex size-20 flex-none items-center justify-center rounded-lg bg-muted text-muted-foreground'>
          {KIND_ICONS[r.kind] ?? <File className='size-5' />}
        </div>
      )}
      <div className='min-w-0 flex-1'>
        <div className='flex items-center gap-2'>
          <span className='truncate font-semibold'>{r.name}</span>
          {via && (
            <Badge variant={VIA_VARIANT[via]} className='flex-none'>
              {VIA_LABEL[via]}
            </Badge>
          )}
          <Badge variant='secondary' className='ml-auto flex-none'>
            {Math.max(0, r.score * 100).toFixed(0)}% match
          </Badge>
        </div>
        <p className='truncate text-xs text-muted-foreground'>{r.path}</p>
        {snippetBody && (
          <p className='mt-1 line-clamp-2 text-sm text-foreground/80'>
            <span className='text-muted-foreground'>{snippetPrefix(r.snippet_source)}</span>
            <MarkedSnippet text={snippetBody} />
          </p>
        )}
        <p className='mt-1 flex items-center gap-1 text-xs text-muted-foreground'>
          <span>
            {fmtBytes(r.size_bytes)} · {fmtDate(r.mtime)}
          </span>
          <Button
            size='sm'
            variant='ghost'
            className='h-6 gap-1 px-2 text-xs text-muted-foreground hover:text-foreground'
            disabled={openState === 'pending'}
            onClick={openFile}
          >
            {openState === 'pending' ? (
              <Loader2 className='size-3 animate-spin' />
            ) : (
              <ExternalLink className='size-3' />
            )}
            {openState === 'error' ? 'Unavailable' : 'Open'}
          </Button>
        </p>
      </div>
    </Card>
  )
}

function kindFallback(kind: string): HTMLElement {
  const div = document.createElement('div')
  div.className = 'flex size-20 flex-none items-center justify-center rounded-lg bg-muted text-muted-foreground'
  div.textContent = kind === 'image' ? '🖼️' : '📄'
  return div
}
