import { createFileRoute } from '@tanstack/react-router'
import { FoldersPage } from '@/features/folders'

export const Route = createFileRoute('/_authenticated/folders/')({
  component: FoldersPage,
})
