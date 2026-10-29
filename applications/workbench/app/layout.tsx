import type {Metadata} from 'next';
import './globals.css';
export const metadata:Metadata={title:'FraudScope | Fraud Operations',description:'A private fraud operations workbench with live scoring, analyst review and temporal graph research on synthetic transactions.',icons:{icon:'/favicon.svg'}};
export default function RootLayout({children}:{children:React.ReactNode}){return <html lang="en"><body>{children}</body></html>;}
