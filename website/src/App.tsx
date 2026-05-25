/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Home from './pages/Home';
import Paper from './pages/Paper';
import Header from './components/Header';
import Footer from './components/Footer';

export default function App() {
  return (
    <Router>
      <div className="min-h-screen text-slate-100 flex flex-col relative w-full selection:bg-[#003cff]/30 selection:text-white bg-[#030303]">
        <Header />
        <div className="flex-1">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/paper" element={<Paper />} />
          </Routes>
        </div>
        <Footer />
      </div>
    </Router>
  );
}

