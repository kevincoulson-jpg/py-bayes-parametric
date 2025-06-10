import socket
import json
import os
import shutil
from FreeCADFEA.vtk_analyze import *
from FreeCADCFD.oF_analyze import *

class FreeCADInterface:
    def __init__(self, ip: str = "127.0.0.1", port: int = 12345, output_json_path: str = "C:/Users/kevin/Documents/py-bayes-parametric/data"):
        self.ip = ip
        self.port = port
        self.commands = []
        self.output_json_path = output_json_path
        self.x_values = {}

        # Initialize required functions
        init_commands = [
            'import FreeCAD',
            'import FreeCADGui',
            'import FemGui',
            'from femtools import ccxtools',
            'import femmesh.gmshtools as gt',
            f'exec(open("C:/Users/kevin/Documents/FreeCAD/Macro/results_json.py").read())',
            f'exec(open("C:/Users/kevin/Documents/FreeCAD/Macro/json_run_and_export.py").read())'
        ]
        self.send_code("\n".join(init_commands))
        
    def add_command(self, command: str):
        """Add a command to the current batch."""
        self.commands.append(command)

    def modify_parameter(self, obj_name: str, param: str, value):
        """Modify a parameter and ensure it's executed in the GUI thread."""
        object_type = self.get_object_type(obj_name)
        if object_type == "<Sketcher::SketchObject>":
            current_value = self.get_param_value(obj_name, param, type='sketch')
            commands = [
                f'object = App.ActiveDocument.getObjectsByLabel("{obj_name}")[0]',
                f'object.setDatum("{param}", {value})',
                'App.activeDocument().recompute()',
                'Gui.updateGui()',  # Force GUI update
                f'print("Parameter {param} updated to {value}")'  # Add verification
            ]
        else:
            current_value = self.get_param_value(obj_name, param, type='other')
            commands = [
                f'object = App.ActiveDocument.getObjectsByLabel("{obj_name}")[0]',
                f'setattr(object, "{param}", {value})',
                'App.activeDocument().recompute()',
                'Gui.updateGui()',  # Force GUI update
                f'print("Parameter {param} updated to {value}")'  # Add verification
            ]

        print(f'current {param} value: {current_value}, updating to {value}')
        result = self.send_code("\n".join(commands))
        print(f"Parameter modification result: {result}")
        return result
    
    def recompute_mesh_fem(self, obj_name: str='FEMMeshGmsh'):
        """Recompute the mesh with proper GUI updates."""
        commands = [
            'import FreeCAD',
            'import FreeCADGui',
            'import femmesh.gmshtools as gt',
            f'object = App.ActiveDocument.getObject("{obj_name}")',
            'mesher = gt.GmshTools(object)',
            'mesher.create_mesh()',
            'App.activeDocument().recompute()',
            'Gui.updateGui()',  # Force GUI update
            'print("Mesh recomputation completed")'  # Add verification
        ]
        result = self.send_code("\n".join(commands))
        print(f"Mesh recomputation result: {result}")
        return result

    def recompute_mesh_cfd(self, obj_name: str='Body_Mesh'):
        """Recompute the mesh with proper GUI updates."""
        commands = [
            'from CfdOF.Mesh import CfdMeshTools',
            'from CfdOF import CfdTools',
            'from CfdOF import CfdConsoleProcess',
            f'cart_mesh = CfdMeshTools.CfdMeshTools(FreeCAD.ActiveDocument.{obj_name})',
            'proxy = FreeCAD.ActiveDocument.Body_Mesh.Proxy',
            'proxy.cart_mesh = cart_mesh',
            'cart_mesh.error = False',
            'cart_mesh.writeMesh()',
            'cmd = CfdTools.makeRunCommand("Allmesh.bat", source_env=False)',
            'env_vars = CfdTools.getRunEnvironment()',
            'proxy.running_from_macro = True',
            'if proxy.running_from_macro:',
            '    mesh_process = CfdConsoleProcess.CfdConsoleProcess()',
            '    mesh_process.start(cmd, env_vars=env_vars, working_dir=cart_mesh.meshCaseDir)',
            '    mesh_process.waitForFinished()',
            'else:',
            '    proxy.mesh_process.start(cmd, env_vars=env_vars, working_dir=cart_mesh.meshCaseDir)'
        ]
        result = self.send_code("\n".join(commands))
        print(f"CFD mesh recomputation result: {result}")
        return result

    def run_fem(self):
        """Run FEM analysis with proper GUI updates."""
        commands = [
            'import FreeCAD',
            'import FreeCADGui',
            'import FemGui',
            'from femtools import ccxtools',
            'FemGui.setActiveAnalysis(App.ActiveDocument.Analysis)',
            'fea = ccxtools.FemToolsCcx()',
            'fea.purge_results()',
            'fea.reset_all()',
            'fea.update_objects()',
            'fea.check_prerequisites()',
            'fea.run()',
            'fea.load_results()',
            'App.activeDocument().recompute()',
            'Gui.updateGui()',  # Force GUI update
            'print("FEM analysis completed")'  # Add verification
        ]
        result = self.send_code("\n".join(commands))
        print(f"FEM analysis result: {result}")
        return result

    def run_cfd(self):
        """Run CFD analysis with proper GUI updates."""
        commands = [
            'from CfdOF.Solve import CfdCaseWriterFoam',
            'FreeCAD.ActiveDocument.CfdSolver.Proxy.case_writer = CfdCaseWriterFoam.CfdCaseWriterFoam(FreeCAD.ActiveDocument.CfdAnalysis)',
            'writer = FreeCAD.ActiveDocument.CfdSolver.Proxy.case_writer',
            'writer.writeCase()',
            'from CfdOF import CfdTools',
            'from CfdOF import CfdConsoleProcess',
            'proxy = FreeCAD.ActiveDocument.CfdSolver.Proxy',
            'proxy.running_from_macro = True',
            'if proxy.running_from_macro:',
            '    analysis_object = FreeCAD.ActiveDocument.CfdAnalysis',
            '    solver_object = FreeCAD.ActiveDocument.CfdSolver',
            '    working_dir = CfdTools.getOutputPath(analysis_object)',
            '    case_name = solver_object.InputCaseName',
            '    solver_directory = os.path.abspath(os.path.join(working_dir, case_name))',
            '    from CfdOF.Solve import CfdRunnableFoam',
            '    solver_runner = CfdRunnableFoam.CfdRunnableFoam(analysis_object, solver_object)',
            '    cmd = solver_runner.getSolverCmd(solver_directory)',
            '    if cmd is not None:',
            '        env_vars = solver_runner.getRunEnvironment()',
            '        solver_process = CfdConsoleProcess.CfdConsoleProcess(stdout_hook=solver_runner.processOutput)',
            '        solver_process.start(cmd, env_vars=env_vars, working_dir=solver_directory)',
            '        solver_process.waitForFinished()'
        ]
        result = self.send_code("\n".join(commands))
        print(f"CFD analysis result: {result}")
        return result

    def export_results(self):
        self.add_command(f'results = App.ActiveDocument.CCX_Results')
        self.add_command(f'export_results(results, r"{self.output_json_path}/results.json")')
        self.send_code()
    
    def export_results_vtu(self):
        # delete current vtu file to ensure it's not erroneously read
        if os.path.exists(f"{self.output_json_path}/results.vtu"):
            os.remove(f"{self.output_json_path}/results.vtu")
        self.add_command(f'import FreeCAD')
        self.add_command(f'import feminout.importVTKResults')
        self.add_command(f'doc = FreeCAD.ActiveDocument')
        self.add_command(f'results_object = doc.getObject("CCX_Results")')
        self.add_command(f'output_filepath = r"{self.output_json_path}/results.vtu"')
        self.add_command(f'feminout.importVTKResults.export([results_object], output_filepath)')
        self.send_code()

    def export_results_of_cfd(self):
        """Export CFD results by copying the case directory to the data folder."""

        # Define source and destination paths
        source_dir = r"C:/Users/kevin/AppData/Local/Temp/case"
        dest_dir = os.path.join(self.output_json_path, "results_openfoam")

        # Remove existing results directory if it exists
        if os.path.exists(dest_dir):
            shutil.rmtree(dest_dir)
            print(f"Removed existing results directory: {dest_dir}")

        # Copy the case directory to the data folder
        try:
            shutil.copytree(source_dir, dest_dir)
            print(f"Successfully copied CFD results to: {dest_dir}")
        except Exception as e:
            print(f"Error copying CFD results: {e}")
            return False

        return True

    ## TODO: Add methods to add vtk attribute and openfoam attribute to the interface
    
    def initiate_vtk_results(self):
        """
        Initiate a VTK results object.
        """
        self.vtk_mesh = analyze_vtk_results(self.output_json_path + '/results.vtu')

    def initiate_openfoam_results(self):
        """
        Initiate an OpenFOAM case.
        """
        self.internal_mesh, self.boundaries = get_mesh(self.output_json_path + '/results_openfoam/pv.foam')

    def export_dxf(self, obj_name: str='Sketch'):
        """
        Export a DXF file of the object.
        """
        MODIFIED_PATH = "C:/Users/kevin/Documents/bayes-foil/data/"
        self.add_command('import importDXF')
        self.add_command(f's = App.ActiveDocument.getObjectsByLabel("{obj_name}")[0]')
        self.add_command(f'importDXF.export(s, r"{MODIFIED_PATH}/foil.dxf")')
        self.send_code()

    def run_and_set_output(self, command: str):
        """
        Run a command and set the output to a JSON file.
        command: the command to run
        """
        self.add_command(f'output = {command}')
        self.add_command('import json')
        self.add_command(f'with open(r"{self.output_json_path}/output.json", "w") as f:')
        self.add_command(f'    json.dump(output, f)')
        self.send_code()

    def get_part_volume(self, obj_name: str):
        """Get the volume of a part."""
        commands = [
            f'object = App.ActiveDocument.getObjectsByLabel("{obj_name}")[0]',
            'volume = object.Shape.Volume',
            'import json',
            f'with open(r"{self.output_json_path}/output.json", "w") as f:',
            '    json.dump(str(volume), f)'
        ]
        self.send_code("\n".join(commands))
        return self.get_output()

    def set_document(self, doc_path: str):
        """Set the active FreeCAD document."""
        commands = [
            f'doc = FreeCAD.openDocument("{doc_path}")',
            'App.setActiveDocument(doc.Name)',
            'Gui.updateGui()'  # Force GUI update
        ]
        self.send_code("\n".join(commands))

    def send_code(self, code: str = None) -> str:
        """Send code to FreeCAD and get response."""
        if code is None:
            code = "\n".join(self.commands)
            self.commands.clear()
            
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((self.ip, self.port))
            sock.sendall(code.encode())
            response = sock.recv(4096).decode()
        return response

    def get_output(self):
        """Get output from the JSON file."""
        try:
            with open(f"{self.output_json_path}/output.json", 'r') as f:
                result = json.load(f)
            return result
        except Exception as e:
            print(f"Error reading output JSON: {e}")
            return None

    def get_object_type(self, obj_name: str):
        """Get the type of an object."""
        commands = [
            f'object = App.ActiveDocument.getObjectsByLabel("{obj_name}")[0]',
            'import json',
            f'with open(r"{self.output_json_path}/output.json", "w") as f:',
            '    json.dump(str(object), f)'
        ]
        self.send_code("\n".join(commands))
        return self.get_output()

    def get_param_value(self, obj_name: str, param: str, type: str = 'sketch'):
        """Get the value of a parameter."""
        commands = [
            f'object = App.ActiveDocument.getObjectsByLabel("{obj_name}")[0]',
            'import json',
            f'with open(r"{self.output_json_path}/output.json", "w") as f:',
            '    json.dump(str(object.getDatum("' + param + '") if "' + type + '" == "sketch" else getattr(object, "' + param + '")), f)'
        ]
        self.send_code("\n".join(commands))
        return self.get_output()

if __name__ == "__main__":
    test = 'full_workflow_fem'  # Change this to run different tests

    if test == 'full_workflow_fem':
        print("Starting full fem workflow test...")
        interface = FreeCADInterface(output_json_path="C:/Users/kevin/Documents/py-bayes-parametric/examples/FreeCADFEA/data")
        print('interface initialized')
        
        # 1. Set the document
        doc_path = "C:/Users/kevin/Documents/py-bayes-parametric/examples/FreeCADFEA/bracket_test.FCStd"
        print("Setting document:", doc_path)
        interface.set_document(doc_path)
        
        # 2. Modify Sketch HoleDiameter
        print("Modifying Brace Width...")
        interface.modify_parameter('MainProfile', 'wBrace', 5)
        
        # 3. Modify Pad Length
        print("Modifying Pad Length...")
        interface.modify_parameter('Pad001', 'Length', 10)
        
        # 4. Recompute mesh
        print("Recomputing mesh...")
        interface.recompute_mesh_fem()
        
        # 5. Run FEM analysis
        print("Running FEM analysis...")
        interface.run_fem()
        
        # 6. Export results
        print("Exporting results...")
        interface.export_results_vtu()

        # 7. Initiate VTK results
        print("Initiating VTK results...")
        interface.initiate_vtk_results()

        print("VTK results initiated successfully")
        print(interface.vtk_mesh)

        print("Full workflow completed successfully")

        # 8. Analyze point data
        point_to_analyze = (2, 0.0, 5.0)
        point_data = analyze_point_data(interface.vtk_mesh, point_to_analyze)

    elif test == 'full_workflow_cfd':
        interface = FreeCADInterface(output_json_path="C:/Users/kevin/Documents/py-bayes-parametric/examples/FreeCADCFD/data")
        print('interface initialized')

        print("Starting full cfd workflow test...")
        # 1. Set the document
        doc_path = "C:/Users/kevin/Documents/py-bayes-parametric/examples/FreeCADCFD/nozzle_cfd.FCStd"
        print("Setting document:", doc_path)
        interface.set_document(doc_path)
        
        # 2. Modify Sketch HoleDiameter
        print("Modifying Throat Radius...")
        interface.modify_parameter('Sketch', 'rThroat', 10)

        # 3. Recompute mesh
        print("Recomputing mesh...")
        interface.recompute_mesh_cfd()

        # 4. Run CFD analysis
        print("Running CFD analysis...")
        interface.run_cfd()

        # 5. Export results
        print("Exporting results...")
        interface.export_results_of_cfd()

        # 6. Initiate OpenFOAM results
        print("Initiating OpenFOAM results...")
        interface.initiate_openfoam_results()

        print("OpenFOAM results initiated successfully")
        print(interface.internal_mesh)

        print("Full workflow completed successfully")
        
        
    # Add more test cases as needed