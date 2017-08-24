using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Autodesk.DesignScript.Runtime;
using Autodesk.DesignScript.Interfaces;
using Autodesk.DesignScript.Geometry;

namespace DynamoCompas
{
    public class DynamoCompas
    {

        /// <summary>
        /// Placeholder for the creation of a network element
        /// </summary>
        /// <param name="mesh"></param>
        /// <returns></returns>
        public static CompasNetworkWrapper Create(Mesh mesh) => new CompasNetworkWrapper(mesh);

        public static CompasNetworkWrapper Smooth(CompasNetworkWrapper network, int iteration)
        {
            
            return null;
        }

        public static CompasNetwork CompasNetworkFromObj(string filePath = null)
        {
            var pySrc =
@"
import sys
sys.path.append(r'C:\Program Files (x86)\IronPython 2.7\Lib')
sys.path.append(r'C:\Users\martinno\repos\compas\src_load')
import compas
from compas.datastructures.network import Network
# import List class to cast the type
from System.Collections.Generic import *

def NetworkFromObject(filepath):

    # import network
    network = Network.from_obj(filepath)

    # extract network vertices
    xyz = [network.vertex_coordinates(key) for key in network.vertices()]
    vertices = List[object]([List[object]([x, y, z]) for x, y, z in xyz])

    # extract network edges
    key_index = network.key_index()
    edges = [(key_index[u], key_index[v]) for u, v in network.edges()]
    edges = List[object]([List[object](ij) for ij in edges])

    return List[object]([network, str(network), vertices, edges])
";
            if (filePath != null || filePath != "")
            {

                // host python and execute script
                var engine = IronPython.Hosting.Python.CreateEngine();
                var scope = engine.CreateScope();
                engine.Execute(pySrc, scope);

                var NetworkFromObject = scope.GetVariable<Func<string, List<object>>>("NetworkFromObject");
                var networkList = NetworkFromObject(filePath);
                return CompasNetwork.Create(networkList[0], networkList[1] as String, networkList[2] as List<object>, networkList[3] as List<object>);
            }
            return null;
        }
    }

}
