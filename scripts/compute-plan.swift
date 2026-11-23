// Inspect planned placement for a compiled Core ML model. Verify actual execution in Instruments.
import CoreML
import Foundation

guard CommandLine.arguments.count == 3 else {
    FileHandle.standardError.write(Data("usage: swift compute-plan.swift model.mlmodelc cpu|gpu|ane|all\n".utf8))
    exit(2)
}
guard #available(macOS 14.4, *) else {
    print("MLComputePlan requires macOS 14.4+")
    exit(1)
}
let units: [String: MLComputeUnits] = ["cpu": .cpuOnly, "gpu": .cpuAndGPU,
                                       "ane": .cpuAndNeuralEngine, "all": .all]
guard let allowed = units[CommandLine.arguments[2]] else {
    print("choose cpu, gpu, ane, or all")
    exit(2)
}
let config = MLModelConfiguration()
config.computeUnits = allowed
let url = URL(fileURLWithPath: CommandLine.arguments[1])
let plan = try await MLComputePlan.load(contentsOf: url, configuration: config)

func walk(_ block: MLModelStructure.Program.Block, path: String) {
    for (index, op) in block.operations.enumerated() {
        let label = "\(path)/\(index) \(op.operatorName)"
        if let usage = plan.deviceUsage(for: op) {
            print("\(label) preferred=\(usage.preferred) supported=\(usage.supported)")
        } else {
            print("\(label) placement unavailable")
        }
        for (i, nested) in op.blocks.enumerated() { walk(nested, path: "\(label)/block\(i)") }
    }
}

switch plan.modelStructure {
case .program(let program):
    for name in program.functions.keys.sorted() { walk(program.functions[name]!.block, path: name) }
case .neuralNetwork(let network):
    for layer in network.layers {
        print("\(layer.name) \(layer.type) preferred=\(plan.deviceUsage(for: layer)?.preferred.description ?? "unavailable")")
    }
default:
    print("Unsupported structure for this inspector. Inspect each pipeline component in Xcode.")
    exit(1)
}
