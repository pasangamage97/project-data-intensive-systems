import React from "react";

const ResultComponent = ({ data }) => {
  if (!data) {
    return <div className="p-4 text-gray-500"></div>;
  }

  return (
    <div className="p-4 bg-gray-100 rounded-xl shadow-md">
      <h3 className="text-lg font-semibold mb-2">Result</h3>
      <pre className="bg-white p-3 rounded-md overflow-auto text-sm text-gray-800 border border-gray-300">
        {JSON.stringify(data, null, 2)}
      </pre>
    </div>
  );
};

export default ResultComponent;
