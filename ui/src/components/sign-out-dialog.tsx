import { ConfirmDialog } from '@/components/confirm-dialog'

interface SignOutDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
}

/** Local Memory has no accounts or sessions — this dialog explains privacy instead. */
export function SignOutDialog({ open, onOpenChange }: SignOutDialogProps) {
  return (
    <ConfirmDialog
      open={open}
      onOpenChange={onOpenChange}
      title='No account needed'
      desc='Local Memory has no accounts, sessions or cloud. Everything runs on this device — your data never leaves this PC.'
      confirmText='Got it'
      handleConfirm={() => onOpenChange(false)}
      className='sm:max-w-sm'
    />
  )
}
