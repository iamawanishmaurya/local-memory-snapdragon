import { createFileRoute } from '@tanstack/react-router'
import { PrivacySettings } from '@/features/settings'

export const Route = createFileRoute('/_authenticated/settings')({
  component: PrivacySettings,
})
