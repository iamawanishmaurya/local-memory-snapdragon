import { useQuery } from '@tanstack/react-query'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Header } from '@/components/layout/header'
import { Main } from '@/components/layout/main'
import { ProfileDropdown } from '@/components/profile-dropdown'
import { ThemeSwitch } from '@/components/theme-switch'
import { Search as GlobalSearch } from '@/components/search'
import { ConfigDrawer } from '@/components/config-drawer'
import { api, fmtBytes, type StatsReport } from '@/lib/api'

function StatCard({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <Card>
      <CardHeader className='pb-2'>
        <CardTitle className='text-sm font-medium text-muted-foreground'>{label}</CardTitle>
      </CardHeader>
      <CardContent>
        <div className='text-2xl font-bold'>{value}</div>
        {sub && <p className='text-xs text-muted-foreground'>{sub}</p>}
      </CardContent>
    </Card>
  )
}

function BarList({
  rows,
  emptyText,
}: {
  rows: { key: string; label: string; count: number; bytes: number }[]
  emptyText: string
}) {
  const max = Math.max(1, ...rows.map((r) => r.bytes))
  if (rows.length === 0) {
    return <p className='text-muted-foreground'>{emptyText}</p>
  }
  return (
    <div className='space-y-2'>
      {rows.map((r) => (
        <div key={r.key} className='space-y-1'>
          <div className='flex items-center justify-between gap-2 text-sm'>
            <span className='min-w-0 truncate'>{r.label}</span>
            <Badge variant='secondary'>
              {r.count} file{r.count === 1 ? '' : 's'} · {fmtBytes(r.bytes)}
            </Badge>
          </div>
          <div className='h-2 w-full overflow-hidden rounded-full bg-muted'>
            <div
              className='h-full rounded-full bg-primary/70'
              style={{ width: `${Math.max(2, (r.bytes / max) * 100)}%` }}
            />
          </div>
        </div>
      ))}
    </div>
  )
}

export function StatisticsPage() {
  const stats = useQuery({ queryKey: ['stats'], queryFn: api.stats, refetchInterval: 10_000 })
  const s: StatsReport | undefined = stats.data

  return (
    <>
      <Header>
        <GlobalSearch className='me-auto' />
        <ThemeSwitch />
        <ConfigDrawer />
        <ProfileDropdown />
      </Header>

      <Main fixed>
        <h1 className='text-2xl font-bold tracking-tight md:text-3xl'>Statistics</h1>
        <p className='text-muted-foreground'>
          Disk usage and index composition — computed live from the local database.
        </p>

        {!s ? (
          <p className='mt-8 text-center text-muted-foreground'>Loading…</p>
        ) : (
          <>
            <div className='mt-4 grid gap-4 sm:grid-cols-2 lg:grid-cols-4'>
              <StatCard
                label='Total on disk'
                value={fmtBytes(s.database.total_bytes)}
                sub='database + thumbnails'
              />
              <StatCard
                label='Index database'
                value={fmtBytes(s.database.index_db_bytes)}
                sub={
                  s.database.wal_bytes + s.database.shm_bytes > 0
                    ? `WAL ${fmtBytes(s.database.wal_bytes)} · SHM ${fmtBytes(s.database.shm_bytes)}`
                    : 'no WAL/SHM right now'
                }
              />
              <StatCard
                label='Thumbnails'
                value={fmtBytes(s.database.thumbnails_bytes)}
                sub='image preview cache'
              />
              <StatCard
                label='Files indexed'
                value={s.files.total.toLocaleString()}
                sub={`${s.index.total_chunks.toLocaleString()} chunks`}
              />
            </div>

            <div className='mt-4 grid gap-4 lg:grid-cols-2'>
              <Card>
                <CardHeader>
                  <CardTitle className='text-base'>Files by type</CardTitle>
                </CardHeader>
                <CardContent>
                  <BarList
                    rows={s.files.by_kind.map((k) => ({
                      key: k.kind,
                      label: k.kind,
                      count: k.count,
                      bytes: k.bytes,
                    }))}
                    emptyText='No files indexed yet.'
                  />
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className='text-base'>Top extensions</CardTitle>
                </CardHeader>
                <CardContent>
                  <BarList
                    rows={s.files.by_extension.map((e) => ({
                      key: e.ext,
                      label: e.ext,
                      count: e.count,
                      bytes: e.bytes,
                    }))}
                    emptyText='No files indexed yet.'
                  />
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className='text-base'>Top folders</CardTitle>
                </CardHeader>
                <CardContent>
                  <BarList
                    rows={s.files.by_folder.map((f) => ({
                      key: f.folder,
                      label: f.folder,
                      count: f.count,
                      bytes: f.bytes,
                    }))}
                    emptyText='No folders watched yet.'
                  />
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className='text-base'>Index</CardTitle>
                </CardHeader>
                <CardContent className='space-y-2 text-sm'>
                  <div className='flex items-center justify-between'>
                    <span className='text-muted-foreground'>Chunks</span>
                    <span>{s.index.total_chunks.toLocaleString()}</span>
                  </div>
                  <div className='flex items-center justify-between'>
                    <span className='text-muted-foreground'>Embedded (vectors)</span>
                    <span>{s.index.embedded_chunks.toLocaleString()}</span>
                  </div>
                  <div className='flex items-center justify-between'>
                    <span className='text-muted-foreground'>Images understood</span>
                    <span>{s.index.images_understood.toLocaleString()}</span>
                  </div>
                  <div className='flex items-center justify-between'>
                    <span className='text-muted-foreground'>Files with OCR text</span>
                    <span>{s.index.ocr_files.toLocaleString()}</span>
                  </div>
                  <div className='flex items-center justify-between'>
                    <span className='text-muted-foreground'>DB pages</span>
                    <span>
                      {s.index.db_page_count.toLocaleString()} × {s.index.db_page_size} B
                    </span>
                  </div>
                  <div className='flex items-center justify-between'>
                    <span className='text-muted-foreground'>Watcher</span>
                    <Badge variant={s.activity.watcher_running ? 'default' : 'secondary'}>
                      {s.activity.watcher_running ? 'running' : 'stopped'}
                    </Badge>
                  </div>
                  <div className='flex items-center justify-between gap-2'>
                    <span className='text-muted-foreground'>Last scan</span>
                    <span className='min-w-0 truncate text-right'>
                      {s.activity.last_scan.last_file || s.activity.last_scan.status}
                    </span>
                  </div>
                </CardContent>
              </Card>
            </div>
          </>
        )}
      </Main>
    </>
  )
}
