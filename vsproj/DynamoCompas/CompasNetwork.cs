using System.Collections.Generic;
using Autodesk.DesignScript.Interfaces;
using Dynamo.Graph.Nodes;
using Autodesk.DesignScript.Runtime;
using System;

/*
 Autodesk ETH Workshop Zurich
 August 2017
 */

namespace Compas.Dynamo.Datastructures
{
    // Add the IsDesignScriptCompatible attribute to ensure
    // that it gets loaded in Dynamo.
    [IsDesignScriptCompatible]
    public class CompasNetwork : IGraphicItem
    {
        #region private members   

        // IronPython mesh object 
        private object pythonNetwork;
        // string representation of this mesh (brought from python)
        private string str;
        // [[x,y,z],[x,y,z],[..]]
        private List<object> vertices;
        private double[][] verticesDouble;

        // [[ptid0, ptid1],[..]]
        private List<object> edgeIndices;
        private int[][] edgesInt;

        private List<object> frames;
        private double[][][] framesFloats = null;

        #endregion

        #region properties

        // Nothing here yet =).

        #endregion

        #region private methods

        public static string GetPackagePath()
        {
            return Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData) + @"\Dynamo\Dynamo Core\1.3\packages\DynamoCompas\";
        }

        #endregion

        #region public methods

        private CompasNetwork(object _pythonNetwork, string stringRepresentation, List<object> _vertices, List<object> _edgeIndices, List<object> _frames)
        {
            str = stringRepresentation;
            pythonNetwork = _pythonNetwork;
            vertices = _vertices;
            edgeIndices = _edgeIndices;
            frames = _frames;

            // parse the vertices to double arrays
            // [[x,y,z],[x,y,z],[..]]
            verticesDouble = new double[vertices.Count][];
            int i = 0;
            foreach (List<object> p in vertices)
            {
                double[] triple = new double[3];
                triple[0] = (double)p[0];
                triple[1] = (double)p[1];
                triple[2] = (double)p[2];
                verticesDouble[i++] = triple;
            }

            // [[ptid0, ptid1],[..]]
            edgesInt = new int[edgeIndices.Count][];
            i = 0;
            foreach (List<object> e in edgeIndices)
            {
                int[] edge = new int[2];
                edge[0] = (int)e[0];
                edge[1] = (int)e[1];
                edgesInt[i++] = edge;
            }

            int j = 0;
            if (frames != null)
            {
                framesFloats = new double[frames.Count][][];
                i = 0;
                foreach (List<object> f in frames)
                {
                    double[][] verts = new double[vertices.Count][];
                    j = 0;
                    foreach (List<object> xyz in f)
                    {
                        double[] t = new double[3];
                        t[0] = (double)xyz[0];
                        t[1] = (double)xyz[1];
                        t[2] = (double)xyz[2];
                        verts[j++] = t;
                    }
                    framesFloats[i++] = verts;
                }
            }


        }

        public static CompasNetwork CompasNetworkFromObj(string filePath = null, string IronPythonPath = @"C:\Program Files (x86)\IronPython 2.7")
        {
            string path = GetPackagePath() + @"bin";

            var pySrc =
@"
import sys
sys.path.append(r'" + IronPythonPath + @"')
sys.path.append(r'" + IronPythonPath + @"\Lib')
sys.path.append(r'" + IronPythonPath + @"\DLLs')
sys.path.append(r'" + path + @"')

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

        public static CompasNetwork Smooth(CompasNetwork network, int iterations = 100, bool shouldAnimate = false, string IronPythonPath = @"C:\Program Files (x86)\IronPython 2.7")
        {



            string path = GetPackagePath() + @"bin";

            var pySrc =
@"
import sys
sys.path.append(r'" + IronPythonPath + @"')
sys.path.append(r'" + IronPythonPath + @"\Lib')
sys.path.append(r'" + IronPythonPath + @"\DLLs')
sys.path.append(r'" + path + @"')

import compas
from compas.datastructures.network import Network
from compas.datastructures.network.algorithms import smooth_network_centroid

# import List class to cast the type
from System.Collections.Generic import *

def SmoothNetwork(network, its, animate=False):

    def callback(network, k, args):
        frames = args[0]
        animate = args[1]
        if animate:
            frames.append([network.vertex_coordinates(key) for key in network.vertices()])
    
    frames = []
    smooth = network.copy()
    smooth_network_centroid(smooth, fixed = smooth.leaves(), kmax = its, callback=callback, callback_args=(frames, animate))

    # extract network vertices
    xyz = [smooth.vertex_coordinates(key) for key in smooth.vertices()]
    vertices = List[object]([List[object]([x, y, z]) for x, y, z in xyz])

    # extract network edges
    key_index = smooth.key_index()
    edges = [(key_index[u], key_index[v]) for u, v in smooth.edges()]
    edges = List[object]([List[object](ij) for ij in edges])

    # extract animation frames
    frames = List[object]([List[object]([List[object](xyz) for xyz in frame]) for frame in frames])

    return List[object]([smooth, str(smooth), vertices, edges, frames])

";

            if (network != null && network is CompasNetwork)
            {
                // host python and execute script
                var engine = IronPython.Hosting.Python.CreateEngine();
                var scope = engine.CreateScope();
                engine.Execute(pySrc, scope);

                var SmoothNetwork = scope.GetVariable<Func<object, int, bool, List<object>>>("SmoothNetwork");
                var networkList = SmoothNetwork(network.ToPythonNetwork(), iterations, shouldAnimate);

                return CompasNetwork.Create(
                    networkList[0], 
                    networkList[1] as String, 
                    networkList[2] as List<object>, 
                    networkList[3] as List<object>, 
                    networkList[4] as List<object>);
            }

            return null;

        }

        //[CanUpdatePeriodically(true)]
        [IsVisibleInDynamoLibrary(false)]
        public static CompasNetwork Create(object pythonMesh, string stringRepresentation, List<object> vertices, List<object> indices, List<object> frames)
        {
            return new CompasNetwork(pythonMesh, stringRepresentation, vertices, indices, frames);
        }

        //[CanUpdatePeriodically(true)]
        [IsVisibleInDynamoLibrary(false)]
        public static CompasNetwork Create(object pythonMesh, string stringRepresentation, List<object> vertices, List<object> indices)
        {
            return new CompasNetwork(pythonMesh, stringRepresentation, vertices, indices, null);
        }

        #endregion

        #region IGraphicItem interface


        /// <summary>
        /// The Tessellate method in the IGraphicItem interface allows
        /// you to specify what is drawn when dynamo's visualization is
        /// updated.
        /// </summary>
        [IsVisibleInDynamoLibrary(false)]
        public void Tessellate(IRenderPackage package, TessellationParameters parameters)
        {
            // Vertices
            if (verticesDouble != null)
            {
                foreach (double[] p in verticesDouble)
                {
                    package.AddPointVertex(p[0], p[1], p[2]);
                    package.AddPointVertexColor(255, 0, 0, 255);
                }
            }

            if (edgesInt != null)
            {
                foreach (int[] ids in edgesInt)
                {
                    double[] p0 = verticesDouble[ids[0]];
                    double[] p1 = verticesDouble[ids[1]];

                    package.AddLineStripVertexColor(0, 255, 0, 255);
                    package.AddLineStripVertex(p0[0], p0[1], p0[2]);

                    package.AddLineStripVertexColor(0, 0, 255, 255);
                    package.AddLineStripVertex(p1[0], p1[1], p1[2]);

                    package.AddLineStripVertexCount(2);
                }
            }

        }

        #endregion

        public object ToPythonNetwork()
        {
            return this.pythonNetwork;
        }

        public object GetFrames()
        {
            return this.frames;
        }

        public object GetFloatFrames()
        {
            return this.framesFloats;
        }

        public override string ToString()
        {
            //return string.Format("{0}", this.str);
            return this.str;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="network"></param>
        /// <returns></returns>
        [MultiReturn(new[] { "PythonMesh", "Vertices", "Edges" })]
        public static Dictionary<string, object> CompasNetworkProperties(CompasNetwork network)
        {
            return new Dictionary<string, object>()
            {
                { "PythonMesh", network.ToPythonNetwork() },
                { "Vertices", 400 },
                { "Edges", 12 }
            };
        }
    }
}
