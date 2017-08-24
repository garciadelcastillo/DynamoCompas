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
    }

}
