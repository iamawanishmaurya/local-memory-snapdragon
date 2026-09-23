import { createFileRoute } from '@tanstack/react-router'
import { StorageHealthPage } from '@/features/storage'

export const Route = createFileRoute('/_authenticated/storage/')({
  component: StorageHealthPage,
})
