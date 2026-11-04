import initRDKitModule, { type MainModule } from '@rdkit/rdkit'
import wasmUrl from '@rdkit/rdkit/RDKit_minimal.wasm?url'

let rdkit: Promise<MainModule> | undefined

// Load the RDKit WASM module once and share it across the app.
export function getRDKit(): Promise<MainModule> {
  rdkit ??= initRDKitModule({ locateFile: () => wasmUrl })
  return rdkit
}
