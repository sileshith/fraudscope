import {DatabaseSync} from 'node:sqlite';
import fs from 'node:fs';
export function database(path=':memory:'){
 const db=new DatabaseSync(path);db.exec(fs.readFileSync('drizzle/0000_thick_the_captain.sql','utf8'));
 function prepare(sql){return {sql,params:[],bind(...params){this.params=params;return this;},async first(){return db.prepare(sql).get(...this.params)??null;},async all(){return {results:db.prepare(sql).all(...this.params)};},async run(){return {meta:db.prepare(sql).run(...this.params)};}};}
 return {prepare,async batch(statements){db.exec('BEGIN IMMEDIATE');try{const results=[];for(const s of statements)results.push({meta:db.prepare(s.sql).run(...s.params)});db.exec('COMMIT');return results;}catch(e){db.exec('ROLLBACK');throw e;}},raw:db,close:()=>db.close()};
}
